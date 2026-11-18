"""Unit tests for the no-assist Spore Spawn left-ledge policy."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from super_metroid.combat.spore_spawn import (
    ROOM_SPORE_SPAWN,
    SEAT_X,
    SEAT_Y,
    VULNERABLE_SPRITEMAPS,
    SporeSpawnStrategy,
    fight_spore_spawn_action,
    mouth_open,
    play_spore_spawn_fight,
    seated,
)
from super_metroid.ram import GameplayPhase, parse_state


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_SPORE_SPAWN,
        "samus_x": SEAT_X,
        "samus_y": SEAT_Y,
        "pose": 29,
        "enemy0_x": 185,
        "enemy0_y": 600,
        "enemy0_hp": 960,
        "enemy0_spritemap": 0xEF3D,
        "missiles": 10,
        "max_missiles": 10,
        "selected_item": 1,
        "num_enemies": 1,
        "health": 199,
        "max_health": 199,
    }
    values.update(overrides)
    return replace(base, **values)


def test_mouth_open_matches_continuous_spritemap_set() -> None:
    assert mouth_open(_state(enemy0_spritemap=0xEEAF))
    assert mouth_open(_state(enemy0_spritemap=0xEF3D))
    from super_metroid.combat.spore_spawn import eye_fully_open

    assert eye_fully_open(_state(enemy0_spritemap=0xEF3D))
    assert not eye_fully_open(_state(enemy0_spritemap=0xEEAF))
    assert not mouth_open(_state(enemy0_spritemap=0x0000))
    assert 0xEF61 in VULNERABLE_SPRITEMAPS


def test_seated_is_morph_in_left_corner() -> None:
    assert seated(_state())
    assert not seated(_state(pose=1))
    assert not seated(_state(samus_x=90, pose=29))
    assert not seated(_state(samus_y=715, pose=29))


def test_seated_closed_eye_idles() -> None:
    state = _state(enemy0_spritemap=0x0001)
    assert fight_spore_spawn_action(state, 0) == ()


def test_seated_open_eye_unmorphs_to_shoot() -> None:
    action = fight_spore_spawn_action(_state(), 0)
    assert action == ("UP",)


def test_empty_missiles_do_not_unmorph_to_fire() -> None:
    state = _state(missiles=0)
    assert fight_spore_spawn_action(state, 0) == ()


def test_floor_action_walks_left_toward_ledge() -> None:
    state = _state(samus_x=140, samus_y=715, pose=1)
    action = fight_spore_spawn_action(state, 0)
    assert "LEFT" in action


def test_zero_hp_returns_idle() -> None:
    assert fight_spore_spawn_action(_state(enemy0_hp=0), 0) == ()


class _Session:
    def __init__(self, state, *, kill_after: int = 8):
        self.state = state
        self.frame = state.frame
        self.actions: list[tuple[object, str]] = []
        self.kill_after = kill_after

    def step(self, action, reason):
        self.actions.append((action, reason))
        self.frame += 1
        updates = {"frame": self.frame, "selected_item": 1}
        if len(self.actions) >= self.kill_after:
            updates["enemy0_hp"] = 0
        self.state = replace(self.state, **updates)
        return self.state


def test_play_loop_reports_defeat_when_hp_hits_zero() -> None:
    session = _Session(_state(pose=1, samus_y=715, samus_x=40), kill_after=6)
    evidence = play_spore_spawn_fight(session, require_boss_bit=False)
    assert evidence.outcome == "spore_spawn_defeated"
    assert evidence.defeat_frame is not None
    assert evidence.final_enemy_hp == 0


def test_list_pickups_reads_f337_header_and_ilist() -> None:
    from super_metroid.combat.spore_spawn import (
        ADDR_PROJ_ID,
        ADDR_PROJ_ILIST,
        ADDR_PROJ_X,
        ADDR_PROJ_Y,
        ILIST_MISSILES,
        PICKUP_MISSILE,
        PICKUP_PROJ_ID,
        list_pickups,
    )

    ram = np.zeros(0x2000, dtype=np.uint8)
    ram[ADDR_PROJ_ID] = PICKUP_PROJ_ID & 0xFF
    ram[ADDR_PROJ_ID + 1] = PICKUP_PROJ_ID >> 8
    ram[ADDR_PROJ_ILIST] = ILIST_MISSILES & 0xFF
    ram[ADDR_PROJ_ILIST + 1] = ILIST_MISSILES >> 8
    ram[ADDR_PROJ_X] = 80
    ram[ADDR_PROJ_Y] = 200

    class _Env:
        def get_ram(self):
            return ram

    found = list_pickups(_Env())
    assert len(found) == 1
    assert found[0].kind == PICKUP_MISSILE
    assert found[0].x == 80
    assert found[0].y == 200


def test_strategy_defaults_are_two_missile_windows() -> None:
    strategy = SporeSpawnStrategy()
    assert strategy.missiles_per_window == 2
    assert strategy.min_missiles_to_fire >= 1
    assert strategy.farm_until >= 2
    assert strategy.max_fight_frames >= 12_000
    assert strategy.fire_x_right >= 11
    assert strategy.fire_x_left <= 4
    assert strategy.fire_enemy_x_min <= 120


def test_under_eye_is_live_x_not_hardcoded_band() -> None:
    from super_metroid.combat.spore_spawn import (
        _fire_target_x,
        _high_right_park,
        under_eye,
    )

    # Window 1: +8 and the +11 flinch shot hit; −11 (x=174) misses.
    assert under_eye(_state(samus_x=193, enemy0_x=185))
    assert under_eye(_state(samus_x=193, enemy0_x=182))
    assert not under_eye(_state(samus_x=174, enemy0_x=185))
    # Later windows park near 142 — the old 180–195 band is a miss.
    assert under_eye(_state(samus_x=142, enemy0_x=142))
    assert under_eye(_state(samus_x=148, enemy0_x=142))
    assert not under_eye(_state(samus_x=193, enemy0_x=142))

    first = _state(enemy0_x=185, enemy0_y=586)
    later = _state(enemy0_x=142, enemy0_y=604)
    assert _high_right_park(first)
    assert not _high_right_park(later)
    assert _fire_target_x(first, 6) == 188
    assert _fire_target_x(later, 6) == 148

    from super_metroid.combat.spore_spawn import in_fire_height

    # 612 vs 586 hits; 622 vs 586 is a stalk miss; 622 vs 604 is a later-park hit.
    assert in_fire_height(_state(samus_y=612, enemy0_y=586))
    assert not in_fire_height(_state(samus_y=622, enemy0_y=586))
    assert in_fire_height(_state(samus_y=622, enemy0_y=604))
    assert not in_fire_height(_state(samus_y=580, enemy0_y=586))
    # Floor park (96, 666): 20px below hits; floor (715) and above miss.
    assert in_fire_height(_state(samus_y=680, enemy0_y=666))
    assert in_fire_height(_state(samus_y=666, enemy0_y=666))
    assert not in_fire_height(_state(samus_y=715, enemy0_y=666))
    assert not in_fire_height(_state(samus_y=657, enemy0_y=666))


def test_floor_bounce_requires_spore_room_and_full_hp() -> None:
    from super_metroid.combat.spore_spawn import play_spore_spawn_floor_bounce

    session = _Session(_state(room_id=0x9D19), kill_after=2)
    try:
        play_spore_spawn_floor_bounce(session)
    except RuntimeError as exc:
        assert "expected room" in str(exc)
    else:
        raise AssertionError("wrong room must fail")

    session = _Session(_state(enemy0_hp=100), kill_after=2)
    try:
        play_spore_spawn_floor_bounce(session)
    except RuntimeError as exc:
        assert "960 HP" in str(exc)
    else:
        raise AssertionError("low HP must fail")


def test_floor_bounce_reports_defeat_when_hp_hits_zero() -> None:
    from super_metroid.combat.spore_spawn import play_spore_spawn_floor_bounce
    from super_metroid.routes.kpdr.brinstar.spore_spawn import SporeSpawnEvidence

    session = _Session(
        _state(samus_x=100, samus_y=720, enemy0_spritemap=0xEF3D),
        kill_after=6,
    )
    fight = play_spore_spawn_floor_bounce(session)
    assert fight.defeat_frame is not None
    assert 0 in fight.observed_hp
    assert fight.peak_hp >= 960
    hop = SporeSpawnEvidence(
        entry_frame=fight.entry_frame,
        activation_frame=fight.activation_frame,
        defeat_frame=fight.defeat_frame,
        exit_frame=fight.defeat_frame + 1,
        peak_hp=fight.peak_hp,
        observed_hp=fight.observed_hp,
        brinstar_boss_bits_before=fight.brinstar_boss_bits_before,
        brinstar_boss_bits_after=0,
        vulnerable_spritemaps=fight.vulnerable_spritemaps,
    )
    assert hop.observed_hp == fight.observed_hp
    assert any(reason == "fight_spore_spawn" for _, reason in session.actions)


@pytest.mark.rom
def test_rom_rollout_lands_one_missile_without_losing_energy() -> None:
    """The natural enter pin must yield real damage and a bit-exact replay."""
    from super_metroid.combat.spore_spawn import play_spore_spawn_rollout
    from super_metroid.dev.common import boot_from_state, make_dev_env
    from super_metroid.paths import GAME_DIR
    from super_metroid.ram import parse_env_state

    class Session:
        def __init__(self, env):
            self.env, self.frame = env, 0
            self.state = parse_env_state(env, mode="nav")

        def step(self, action, reason):
            self.env.step(action)
            self.frame += 1
            self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
            return self.state

    env = make_dev_env()
    try:
        boot_from_state(env, GAME_DIR / "tasks/full_start_v1_anchors/"
                        "f015374_enter_0x9DC7_0x9DC7.state", settle_frames=0)
        session = Session(env)
        before = session.state
        report = play_spore_spawn_rollout(
            session, workers=4, target_damage=100, max_frames=600,
        )
        assert report["outcome"] == "damage_window"
        assert session.state.health == before.health
        assert session.state.enemy0_hp <= before.enemy0_hp - 100
        assert session.state.missiles == before.missiles - 1
        assert report["rollout_restores"] > 0
    finally:
        env.close()


@pytest.mark.rom
@pytest.mark.parametrize(
    ("pin_name", "energy_spent", "missiles_spent"),
    [
        ("spore_rollout_hp60_seat_clean.state", 0, 1),
        ("scratch/spore_master_hp160_clean.state", 12, 0),
    ],
    ids=["human-hit-free", "master-energy-retry"],
)
def test_rom_rollout_finishes_kill_and_exit_to_super_room(
    pin_name: str, energy_spent: int, missiles_spent: int,
) -> None:
    """The late fight clears from the human and the natural master entries."""
    from super_metroid.dev.common import boot_from_state, make_dev_env
    from super_metroid.paths import INTEGRATION_DIR
    from super_metroid.ram import parse_env_state
    from super_metroid.routes.kpdr.brinstar.spore_spawn import (
        play_clean_spore_spawn_leave,
    )

    pin = INTEGRATION_DIR / pin_name
    if not pin.is_file():
        pytest.skip("development safe-seat pin not present in clone")

    class Session:
        def __init__(self, env):
            self.env, self.frame = env, 0
            self.state = parse_env_state(env, mode="nav")

        def step(self, action, reason):
            self.env.step(action)
            self.frame += 1
            self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
            return self.state

    env = make_dev_env()
    try:
        boot_from_state(env, pin, settle_frames=0)
        session = Session(env)
        before = session.state
        result = play_clean_spore_spawn_leave(session)
        assert result.report["outcome"] == "spore_spawn_defeated"
        assert session.state.room_id == 0x9B5B
        assert session.state.game_state == 8
        assert not session.state.door_transition
        assert session.state.health == before.health - energy_spent
        assert session.state.missiles == before.missiles - missiles_spent
        assert parse_env_state(env).boss_bits[1] & 0x02
        assert result.fight.rollout_restores > 0
    finally:
        env.close()
