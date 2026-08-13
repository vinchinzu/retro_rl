"""Unit tests for the no-assist Spore Spawn left-ledge policy."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.combat.protocol import wrap_spore_spawn_as_boss_strategy
from super_metroid.combat.spore_spawn import (
    ROOM_SPORE_SPAWN,
    SEAT_X,
    SEAT_Y,
    VULNERABLE_SPRITEMAPS,
    SporeSpawnEvidence,
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


def test_evidence_to_dict_keys_are_stable() -> None:
    evidence = SporeSpawnEvidence(
        start_frame=10,
        activation_seen=True,
        defeat_frame=200,
        boss_bit_frame=240,
        end_frame=250,
        peak_hp=960,
        min_enemy_hp=0,
        action_frames=240,
        final_enemy_hp=0,
        shots_fired=10,
        farm_frames=40,
        windows=5,
        outcome="spore_spawn_defeated",
        vulnerable_spritemaps=(0xEEAF,),
    )
    assert set(evidence.to_dict()) == {
        "start_frame",
        "activation_seen",
        "defeat_frame",
        "boss_bit_frame",
        "end_frame",
        "peak_hp",
        "min_enemy_hp",
        "action_frames",
        "final_enemy_hp",
        "shots_fired",
        "farm_frames",
        "windows",
        "outcome",
        "vulnerable_spritemaps",
    }


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


def test_wrapper_matches_catalog_room() -> None:
    strategy = wrap_spore_spawn_as_boss_strategy()
    assert strategy.boss_id == "spore_spawn"
    assert strategy.catalog.room_id == ROOM_SPORE_SPAWN
    assert strategy.entry.room_id == ROOM_SPORE_SPAWN


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
