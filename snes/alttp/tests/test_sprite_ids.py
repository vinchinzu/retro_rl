"""Sprite type names, including the eight verified primitives constants."""

from __future__ import annotations

import json

import pytest

from alttp.primitives import (
    SPRITE_BALL_AND_CHAIN,
    SPRITE_BIG_KEY,
    SPRITE_BLUE_SOLDIER,
    SPRITE_GREEN_SOLDIER,
    SPRITE_HEART,
    SPRITE_KNIGHT,
    SPRITE_SMALL_KEY,
    SPRITE_ZELDA,
)
from alttp.sprite_ids import (
    EFFECT_BURN,
    EFFECT_FREEZE,
    EFFECT_STUN_255,
    SPRITE_NAMES,
    VULNERABILITY,
    sprite_name,
    vulnerability,
)
from alttp.z3_json_data import Z3JsonData, default_data_root

# z3-json-data parks dark-world stat aliases at these ids. They are not
# sprite type bytes (Dacto is 216, while 0xD8 is the heart pickup).
_Z3_ALIAS_ID = 216

_VERIFIED = (
    (SPRITE_BLUE_SOLDIER, 0x41, "blue soldier"),
    (SPRITE_GREEN_SOLDIER, 0x42, "green soldier"),
    (SPRITE_KNIGHT, 0x4B, "knight"),
    (SPRITE_BALL_AND_CHAIN, 0x6A, "ball and chain"),
    (SPRITE_ZELDA, 0x76, "zelda"),
    (SPRITE_HEART, 0xD8, "heart"),
    (SPRITE_SMALL_KEY, 0xE4, "small key"),
    (SPRITE_BIG_KEY, 0xE5, "big key"),
)


def test_verified_eight_names() -> None:
    assert len(_VERIFIED) == 8
    for constant, sprite_type, name in _VERIFIED:
        assert constant == sprite_type
        assert sprite_name(sprite_type) == name
        assert SPRITE_NAMES[sprite_type] == name


def test_sprite_type_table_covers_routines() -> None:
    assert len(SPRITE_NAMES) == 0xF3
    assert set(SPRITE_NAMES) == set(range(0xF3))
    assert all(name for name in SPRITE_NAMES.values())
    assert sprite_name(0x03) == "unused"
    assert sprite_name(0xF2) == "medallion tablet"
    assert sprite_name(0xF3) == "unknown"
    assert sprite_name(-1) == "unknown"


def test_bosses_followers_and_pickups() -> None:
    assert sprite_name(0x7A) == "agahnim"
    assert sprite_name(0x53) == "armos knight"
    assert sprite_name(0x88) == "mothula"
    assert sprite_name(0xD6) == "ganon"
    assert sprite_name(0xAD) == "old man"
    assert sprite_name(0xAB) == "crystal maiden"
    assert sprite_name(0xB6) == "kiki"
    assert sprite_name(0xB7) == "blind maiden"
    assert sprite_name(0xD9) == "green rupee"
    assert sprite_name(0xE3) == "fairy"
    assert sprite_name(0xEA) == "heart container"
    assert sprite_name(0xEB) == "heart piece"
    assert sprite_name(0xEE) == "mantle"


def test_z3_enemy_ids_below_alias_block_are_named() -> None:
    data = Z3JsonData.load()
    named = [enemy for enemy in data.enemies if enemy.id < _Z3_ALIAS_ID and any(enemy.names)]
    assert len(named) >= 100
    for enemy in named:
        assert sprite_name(enemy.id) != "unknown"
    assert sprite_name(0xD8) == "heart"


# Castle guard block. Ball-and-chain (0x6A) and the cannon soldier (0x6B)
# match the same way but are not in this range.
_CASTLE_SOLDIERS = range(0x41, 0x4C)


def test_castle_soldiers_verified_against_z3() -> None:
    payload = json.loads((default_data_root() / "enemies" / "main.json").read_text())
    by_id = {enemy["id"]: enemy for enemy in payload["enemies"] if enemy["id"] < _Z3_ALIAS_ID}
    assert len(VULNERABILITY) == 0xF3
    blue = vulnerability(0x41)
    assert blue.verified and not blue.ignores_damage
    assert blue.hp == 6
    assert blue.contact_class == 1
    assert blue.contact_damage == (4, 4, 4)
    assert blue.effect("sword") == 2
    assert blue.effect("arrow") == 4
    assert blue.effect("bomb") == 4
    assert blue.effect("hammer") == 8
    assert blue.effect("fire") == EFFECT_BURN
    assert blue.effect("ice") == 8
    assert blue.effect("hookshot") == EFFECT_STUN_255
    assert blue.hookshot_stun
    assert blue.hurts("sword") and not blue.hurts("hookshot")
    green = vulnerability(0x42)
    assert green.verified and green.hp == 4
    assert green.effect("ice") == EFFECT_FREEZE
    assert green.hookshot_stun
    for sprite_type in _CASTLE_SOLDIERS:
        row = vulnerability(sprite_type)
        enemy = by_id[sprite_type]
        assert row.verified, hex(sprite_type)
        assert not row.ignores_damage
        assert row.hp == enemy["hp"]
        assert list(row.contact_damage) == enemy["dmgToLink"]
        assert list(row.damage) == enemy["dmgFromLink"]
        assert row.hurts("sword")
        assert row.hurts("arrow")
        assert row.hurts("bomb")
        assert row.hurts("hammer")
        assert row.effect("fire") == EFFECT_BURN
        assert row.hookshot_stun
    for sprite_type in (0x6A, 0x6B):
        row = vulnerability(sprite_type)
        enemy = by_id[sprite_type]
        assert row.verified
        assert row.hp == enemy["hp"]
        assert list(row.damage) == enemy["dmgFromLink"]
    # ROM init HP is 12; z3 says 4. Give-damage matched, so the row stays unverified.
    assert not vulnerability(0x00).verified
    assert vulnerability(0x00).hp == 12
    assert by_id[0x00]["hp"] == 4
    heart = vulnerability(0xD8)
    assert heart.ignores_damage and not heart.verified
    assert not heart.hurts("sword")
    with pytest.raises(TypeError):
        vulnerability(True)
    with pytest.raises(ValueError):
        vulnerability(0xF3)
    with pytest.raises(ValueError):
        blue.effect("boomerang")
