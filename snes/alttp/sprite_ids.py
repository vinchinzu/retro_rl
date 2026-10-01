"""Sprite type byte → short name.

The type byte is WRAM ``$0E20`` (``primitives.SPRITE_TYPE``). Names follow
the snesrev zelda3 ``kSpriteActiveRoutines`` table, types ``0x00``–``0xF2``.
A ``z3-json-data`` ``enemies/main.json`` row is used when its ``id`` is that
same sprite type. Ids ``>= 216`` in that file are shared-stat aliases
(Dacto is 216), not sprite types: ``0xD8`` is the heart pickup.

Eight ids stay the verified ``primitives.py`` constants: ``0x41`` blue
soldier, ``0x42`` green soldier, ``0x4B`` knight, ``0x6A`` ball and chain,
``0x76`` zelda, ``0xD8`` heart, ``0xE4`` small key, ``0xE5`` big key.
``0x4B`` is ``Sprite_4B_GreenKnifeGuard`` in the disassembly.

Vulnerability rows are ROM give-damage bytes: subclass nibbles decompressed
from ``$83:E800`` (sprites ``0x00``–``0xD7``) indexed through ``kEnemyDamages``
at ``$0D:B8F1``. HP is ``kSpriteInit_Health`` at ``$0D:B173``. Contact class
is the low nibble of ``kSpriteInit_BumpDamage`` at ``$0D:B266``; the green /
blue / red mail triple is ``kPlayerDamages`` at ``$06:F427``. ``verified``
means HP, that triple, and all 16 give-damage bytes match the z3-json-data
enemy with the same id. Castle soldiers ``0x41``–``0x4B`` match. Seven ids
disagree on HP or contact and stay unverified: ``0x00``, ``0x08``, ``0x0A``,
``0x26``, ``0x86``, ``0xC9``, ``0xD7`` (their give-damage bytes did match).
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass

# Index is the sprite type byte. ``0x03`` has a NULL routine.
_NAMES: tuple[str, ...] = (
    "raven",
    "vulture",
    "stalfos head",
    "unused",
    "pull switch down",
    "pull switch up",
    "pull switch down",
    "pull switch up",
    "octorok",
    "moldorm",
    "octorok 4-way",
    "cucco",
    "octorok rock",
    "buzzblob",
    "snapdragon",
    "octoballoon",
    "octoballoon baby",
    "hinox",
    "moblin",
    "mini helmasaur",
    "thieves town grate",
    "antifairy",
    "elder",
    "hoarder bush",
    "mini moldorm",
    "poe",
    "smithy",
    "enemy arrow",
    "statue",
    "flute quest",
    "crystal switch",
    "sick kid",
    "sluggula",
    "water switch",
    "ropa",
    "red bari",
    "blue bari",
    "talking tree",
    "hardhat beetle",
    "deadrock",
    "hint npc",
    "villager",
    "sweeping lady",
    "hobo",
    "lumberjacks",
    "telepathic tile",
    "flute boy",
    "maze game lady",
    "maze game guy",
    "fortune teller",
    "quarrel brothers",
    "rupee pull",
    "young snitch",
    "innkeeper",
    "witch",
    "waterfall",
    "eye statue",
    "locksmith",
    "magic bat",
    "dash item",
    "trough boy",
    "old snitch",
    "hoarder rock",
    "tutorial guard",
    "lightning lock",
    "blue soldier",
    "green soldier",
    "red spear soldier",
    "assault soldier",
    "green spear soldier",
    "blue archer",
    "green archer",
    "red javelin soldier",
    "red bush guard",
    "bomb soldier",
    "knight",
    "geldman",
    "toppo",
    "popo",
    "popo slow",
    "cannonball",
    "armos",
    "king zora",
    "armos knight",
    "lanmolas",
    "zora",
    "walking zora",
    "desert statue",
    "crab",
    "lost woods bird",
    "lost woods squirrel",
    "spark clockwise",
    "spark counterclockwise",
    "roller down",
    "roller up",
    "roller right",
    "roller left",
    "beamos",
    "master sword",
    "debirando pit",
    "debirando",
    "archery game",
    "wall cannon right",
    "wall cannon left",
    "wall cannon down",
    "wall cannon up",
    "ball and chain",
    "cannon soldier",
    "mirror portal",
    "rat",
    "rope",
    "keese",
    "helmasaur fireball",
    "leever",
    "fairy pond",
    "uncle and priest",
    "running man",
    "bottle vendor",
    "zelda",
    "antifairy",
    "mrs sahasrahla",
    "bee",
    "agahnim",
    "agahnim ball",
    "green stalfos",
    "big spike",
    "firebar clockwise",
    "firebar counterclockwise",
    "firesnake",
    "water tektite",
    "antifairy circle",
    "green eyegore",
    "red eyegore",
    "yellow stalfos",
    "kodongo",
    "kodongo fire",
    "mothula",
    "mothula beam",
    "spike block",
    "gibdo",
    "arrghus",
    "arrghus puff",
    "terrorpin",
    "zol",
    "wallmaster",
    "stalfos knight",
    "helmasaur king",
    "bumper",
    "pirogusu",
    "laser eye right",
    "laser eye left",
    "laser eye down",
    "laser eye up",
    "pengator",
    "kyameron",
    "wizzrobe",
    "zoro",
    "babusu",
    "haunted grove ostrich",
    "haunted grove rabbit",
    "haunted grove bird",
    "freezor",
    "kholdstare",
    "kholdstare shell",
    "falling ice",
    "blue zazak",
    "red zazak",
    "stalfos",
    "green zirro",
    "blue zirro",
    "pikit",
    "crystal maiden",
    "apple",
    "old man",
    "pipe down",
    "pipe up",
    "pipe left",
    "pipe right",
    "good bee",
    "pedestal plaque",
    "purple chest",
    "bomb shop",
    "kiki",
    "blind maiden",
    "dialogue tester",
    "bully",
    "whirlpool",
    "shopkeeper",
    "drunkard",
    "vitreous",
    "vitreous eye",
    "lightning",
    "catfish",
    "cutscene agahnim",
    "boulder",
    "gibo",
    "thief",
    "medusa",
    "four way shooter",
    "pokey",
    "big fairy",
    "tektite",
    "chain chomp",
    "trinexx",
    "trinexx fire head",
    "trinexx ice head",
    "blind",
    "swamola",
    "lynel",
    "bunny beam",
    "flopping fish",
    "stal",
    "landmine",
    "digging game guy",
    "ganon",
    "ganon stunned",
    "heart",
    "green rupee",
    "blue rupee",
    "red rupee",
    "bomb",
    "bombs 4",
    "bombs 8",
    "small magic",
    "full magic",
    "arrows 5",
    "arrows 10",
    "fairy",
    "small key",
    "big key",
    "shield",
    "mushroom",
    "fake sword",
    "potion shop",
    "heart container",
    "heart piece",
    "thrown item",
    "somaria platform",
    "mantle",
    "somaria platform",
    "somaria platform",
    "somaria platform",
    "medallion tablet",
)

SPRITE_NAMES: Mapping[int, str] = {index: name for index, name in enumerate(_NAMES)}


def sprite_name(sprite_type: int) -> str:
    """Short name for a sprite type byte, or ``unknown`` past ``0xF2``."""
    if isinstance(sprite_type, bool):
        raise TypeError(f"sprite type must be an int, got {sprite_type!r}")
    try:
        index = operator.index(sprite_type)
    except TypeError:
        raise TypeError(f"sprite type must be an int, got {sprite_type!r}") from None
    if 0 <= index < len(_NAMES):
        return _NAMES[index]
    return "unknown"


# Damage-class index into ``SpriteVulnerability.damage``.
# Hammer is class 3, the same row as a tempered slash. Fighter slash is class 1.
# Types ``>= 0xD8`` never reach this table.
CLASS_BOOMERANG = 0
CLASS_SWORD = 1
CLASS_HAMMER = 3
CLASS_ARROW = 6
CLASS_HOOKSHOT = 7
CLASS_BOMB = 8
CLASS_FIRE = 11
CLASS_ICE = 12

# ``sprite_give_damage`` bytes that are not HP subtraction.
EFFECT_FAIRY = 0xF9
EFFECT_BLOB = 0xFA
EFFECT_STUN_32 = 0xFB
EFFECT_STUN_128 = 0xFC
EFFECT_BURN = 0xFD
EFFECT_FREEZE = 0xFE
EFFECT_STUN_255 = 0xFF
_STUN_EFFECTS = frozenset({EFFECT_STUN_32, EFFECT_STUN_128, EFFECT_STUN_255})

# ``link_give_damage`` for contact classes 0..9: green, blue, red mail.
CONTACT_BY_MAIL: tuple[tuple[int, int, int], ...] = (
    (2, 1, 1),
    (4, 4, 4),
    (0, 0, 0),
    (8, 4, 2),
    (8, 8, 8),
    (16, 8, 4),
    (32, 16, 8),
    (32, 24, 16),
    (24, 16, 8),
    (64, 48, 24),
)

_WEAPON_CLASS = {
    "sword": CLASS_SWORD,
    "arrow": CLASS_ARROW,
    "bomb": CLASS_BOMB,
    "fire": CLASS_FIRE,
    "ice": CLASS_ICE,
    "hammer": CLASS_HAMMER,
    "hookshot": CLASS_HOOKSHOT,
}

# hp, contact class, ignores_damage, verified, 16 give-damage bytes.
# ignores_damage is flags3 bit 6, or any type >= 0xD8 (no subclass row).
_VULN_PACKED: tuple[str, ...] = (
    "12,3,0,0,0102040810100440046400fdfefdfe20",
    "6,3,0,1,0102040810100440046400fdfefdfe20",
    "255,1,0,1,01020408101000000000000000000000",
    "3,2,1,0,00000000000000000000000000000000",
    "3,2,1,0,00000000000000000000000000000000",
    "3,2,1,0,00000000000000000000000000000000",
    "3,2,1,0,00000000000000000000000000000000",
    "3,2,1,0,00000000000000000000000000000000",
    "2,1,0,0,ff020408101004ff0464fafdfefdfefa",
    "12,3,0,1,00020204080804ff0464000808000000",
    "4,1,0,0,ff020408101004ff0464fafdfefdfefa",
    "255,1,0,1,01020408101004400464000808000000",
    "0,1,0,1,01020408101004400464000808101020",
    "3,1,0,1,ff020408101004ff0464f9fdfefdfeff",
    "12,8,0,1,ff020408101004ff0464fafdfefdfefa",
    "2,1,0,1,0102040810100440046400fd08fd1040",
    "0,1,0,1,01020408101004ff0464000808101020",
    "20,8,0,1,fc0204081040040040640008fefdfefa",
    "4,5,0,1,ff020408101004ff4064fafdfefdfefa",
    "4,3,0,1,00020408101004ff4064fa0000fdfefa",
    "0,0,1,0,00000000000000000000000000000000",
    "255,4,0,1,00000000000000000000f90000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "2,2,0,1,01020408101004ff0464fa40404040fa",
    "3,3,0,1,00020408101004400464fafd08fd10fa",
    "8,5,0,1,0102044010100440046400fd08fd40ff",
    "0,0,1,0,00000000000000000000000000000000",
    "0,1,0,1,20404040404040404018004040401040",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,0,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "8,6,0,1,ff020408101004ff0064fafdfefdfefa",
    "3,0,1,0,00000000000000000000000000000000",
    "8,5,0,1,ff020408101004ff0464fafdfefdfefa",
    "2,3,0,1,0002040810100440046400fd40fd40ff",
    "2,1,0,1,0002040810100440046400fd40fd40ff",
    "0,0,1,0,00000000000000000000000000000000",
    "3,0,0,0,00020408101000ffff64000000fdfeff",
    "255,3,0,1,00000000000000000000fa00000000fa",
    "0,0,1,0,00000000000000000000000000000000",
    "3,0,1,0,00000000000000000000000000000000",
    "3,0,0,0,00000000000000000000f90000000000",
    "3,0,1,0,00000000000000000000000000000000",
    "3,0,1,0,00000000000000000000000000000000",
    "3,0,1,0,00000000000000000000000000000000",
    "3,0,1,0,00000000000000000000000000000000",
    "3,0,1,0,00000000000000000000000000000000",
    "3,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "3,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "3,0,1,0,00000000000000000000000000000000",
    "3,0,1,0,00000000000000000000000000000000",
    "3,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "3,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "3,0,1,0,00000000000000000000000000000000",
    "2,2,0,1,01020408101004ff0464fa08404040fa",
    "255,2,1,0,00004008101004ff046400080810fe20",
    "2,0,0,1,00000408101000000000000000000000",
    "6,1,0,1,ff020408101004ff046400fd08fdfefa",
    "4,1,0,1,ff020408101004ff0464fafdfefdfefa",
    "8,3,0,1,ff020304101004ff046400fd00fdfefa",
    "6,1,0,1,ff020408101004ff046400fd00fdfefa",
    "8,3,0,1,ff020304101004ff046400fd00fdfefa",
    "6,1,0,1,ff020408101004ff046400fdfefdfefa",
    "4,1,0,1,ff020408101004ff046400fd08fd40fa",
    "8,3,0,1,ff020304101004ff046400fd00fdfefa",
    "8,3,0,1,ff020304101004ff046400fd00fd40fa",
    "8,3,0,1,ff020304101004ff046400fdfefdfefa",
    "4,1,0,1,ff020408101004ff0464fafdfefdfefa",
    "4,3,0,1,01020408101040ff04640040084010ff",
    "2,1,0,1,01020408101004ff046400fd40fd40ff",
    "2,1,0,1,ff020408101004400464fafdfefdfefa",
    "2,1,0,1,ff020408101004400464fafdfefdfefa",
    "255,1,1,1,00000000000000000000000000000000",
    "8,1,1,1,ff020408101040ff04640008fefd10ff",
    "255,1,1,0,00000000000000000000000000000000",
    "48,1,1,1,01040204080810000464000808000000",
    "16,4,0,1,00020204080804000464000808000000",
    "8,1,0,1,01020408101004ff046400fd08fd4020",
    "8,1,0,1,ff020408101004ff046400fd08fd4020",
    "255,2,1,0,00000000000000000000000000000000",
    "2,5,0,1,ff020408101004ff0464fa08fefdfefa",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "255,4,1,1,00000000000000000000000000fd00ff",
    "255,4,1,1,00000000000000000000000000fd00ff",
    "255,8,1,1,00000000000000000000000000000000",
    "255,8,1,1,00000000000000000000000000000000",
    "255,8,1,1,00000000000000000000000000000000",
    "255,8,1,1,00000000000000000000000000000000",
    "255,4,1,1,00000000000000000000000000000000",
    "255,0,1,0,00000000000000000000000000000000",
    "255,4,0,1,00000000000000000000000000000000",
    "4,3,0,1,01020408101004400464004008401040",
    "4,2,1,0,00000000000000000000000000000000",
    "255,2,1,0,00000000000000000000000000000000",
    "255,2,1,0,00000000000000000000000000000000",
    "255,2,1,0,00000000000000000000000000000000",
    "255,2,1,0,00000000000000000000000000000000",
    "16,3,0,1,fb020208101004ff046400fd00fdfefa",
    "3,1,0,1,ff020408101004ff046400080810fe20",
    "0,0,1,0,00000000000000000000000000000000",
    "2,0,0,1,ff020408101004400464fafdfefdfefa",
    "4,1,0,1,ff020408101004400464fafdfefdfefa",
    "1,0,0,1,0102040810100440046400fdfefdfeff",
    "255,5,0,1,00000000000000000000000000000000",
    "4,1,0,1,ff020408101004ff046400fdfefdfe40",
    "255,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "255,4,1,1,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,0,1,01020408101004400464f90808101020",
    "96,4,0,1,00000000000000000000000000000000",
    "255,4,1,1,00000000000000000000000000000000",
    "24,6,0,1,0002040810100400406400fdfefd10ff",
    "255,4,1,1,00000000000000000000000000000000",
    "255,4,1,1,00000000000000000000000000000000",
    "255,4,1,1,00000000000000000000000000000000",
    "3,4,1,1,00000000000000000000000000000000",
    "4,3,0,1,0002040810100440046400fdfefd1040",
    "255,4,1,1,00000000000000000000000000000000",
    "16,4,0,1,00020440401040000418000000000000",
    "8,4,0,1,00000000000004000064000000000000",
    "8,1,0,1,01020408101004ff0464fafd40fd40fa",
    "0,4,0,0,0002040810100400006400fdfefdfefa",
    "255,4,1,1,00000000000000000000000000000000",
    "32,5,0,1,00020408000000000000000400000000",
    "32,5,1,1,00000000000000000000000000000000",
    "32,4,0,1,00000000000000000000000000000000",
    "32,5,0,1,ff02040810100000046400fdfefdfeff",
    "32,5,1,1,00000408101004000064000808000000",
    "8,5,1,1,00000408101000000464000808000000",
    "8,3,0,1,01020408101004ff40640008fe40feff",
    "4,5,0,1,ff02040810100440406400fdfefdfe40",
    "8,0,0,1,0102040810100400046400fd08fd1040",
    "64,5,0,1,01020408101000000000000000000000",
    "48,5,0,1,00000408101004000464000000000000",
    "255,5,0,0,00000000000000000000000000000000",
    "2,5,0,1,0102040810100440046400fd08fd40ff",
    "255,6,0,1,00000000000000000000000000000000",
    "255,6,0,1,00000000000000000000000000000000",
    "255,6,0,1,00000000000000000000000000000000",
    "255,6,0,1,00000000000000000000000000000000",
    "16,5,0,1,01020408101004404064000800fdfe40",
    "4,3,0,1,010204081010044004640040fe40fe20",
    "2,6,0,1,00020408101004000064000808401040",
    "4,5,0,1,0002040810100440046400fd40fd4040",
    "4,5,0,1,0002040810100440046400fd40fd4040",
    "8,3,0,0,00000000000000000000000000000000",
    "8,3,0,0,00000000000000000000000000000000",
    "8,3,0,0,00000000000000000000000000000000",
    "16,6,0,1,0000000010100000000000fd00fd0000",
    "64,7,0,1,00000408101000000000001000000000",
    "64,5,0,1,00000000000000000000000800400000",
    "8,5,1,1,0002040810100400046400fd00101020",
    "4,5,0,1,ff020408101004ff406400fdfefdfeff",
    "8,5,0,1,ff020408101004ff406400fdfefdfeff",
    "4,1,0,1,01020408101004ff0464fafd40fd40fa",
    "4,5,0,1,0002040810100440046400fdfefdfe20",
    "8,3,0,1,0002040810100440046400fdfefdfe20",
    "12,5,0,1,01020408101004ff046400fd08fdfeff",
    "16,4,1,0,01020408101004ff0464000808000000",
    "0,0,0,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,1,01020408101004400464000808101020",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "128,7,0,1,00000408101010000464000000000000",
    "48,7,0,1,00000408101010000464000000000000",
    "255,5,0,1,00000000000000000000000000000000",
    "255,5,1,0,00000000000000000000000000000000",
    "255,5,0,0,00000000000000000000000000000000",
    "255,4,1,1,00000000000000000000000000000000",
    "8,3,1,1,00020408101010000464000000000000",
    "0,2,0,0,00000000000000000000000000000000",
    "0,0,0,0,00000000000000000000000000101020",
    "0,0,0,0,00000000000000000000000000101020",
    "32,6,0,1,0002040810100400041800fd08fdfeff",
    "0,0,1,0,00000000000000000000000000000000",
    "8,5,0,0,fb02040810101000406400fdfefdfefa",
    "5,7,1,1,fb000000000000000000000000000000",
    "40,7,1,1,00000408101000000000000000000000",
    "40,7,0,1,00000408101000000000000004000000",
    "40,7,0,1,00000408101000000000000400000000",
    "90,5,0,1,00020408101000000000000000000000",
    "16,7,0,1,01020408101004400464000808104020",
    "24,6,0,1,00000008101000000018000000000000",
    "64,0,0,1,00000000000000000000f90000404040",
    "0,0,1,0,00000000000000000000fa00000000fa",
    "4,3,1,1,01020408101004ff0464fa08081040fa",
    "0,3,1,1,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "255,9,0,1,00000004080800000000000000000000",
    "255,9,0,0,00000000000000000018000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
    "0,0,1,0,00000000000000000000000000000000",
)


def _load_vulnerability() -> tuple["SpriteVulnerability", ...]:
    rows: list[SpriteVulnerability] = []
    if len(_VULN_PACKED) != 0xF3:
        raise RuntimeError(f"vulnerability pack has {len(_VULN_PACKED)} rows, expected {0xF3}")
    for packed in _VULN_PACKED:
        hp_s, class_s, ignore_s, verified_s, hex_damage = packed.split(",")
        damage = bytes.fromhex(hex_damage)
        if len(damage) != 16:
            raise RuntimeError("give-damage row is not 16 bytes")
        contact_class = int(class_s)
        if not 0 <= contact_class < len(CONTACT_BY_MAIL):
            raise RuntimeError(f"contact class {contact_class} out of range")
        rows.append(
            SpriteVulnerability(
                hp=int(hp_s),
                contact_class=contact_class,
                damage=tuple(damage),
                ignores_damage=bool(int(ignore_s)),
                verified=bool(int(verified_s)),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class SpriteVulnerability:
    """HP, contact class, and the 16 give-damage bytes for one sprite type."""

    hp: int
    contact_class: int
    damage: tuple[int, ...]
    ignores_damage: bool
    verified: bool

    @property
    def contact_damage(self) -> tuple[int, int, int]:
        """Green, blue, then red mail, in ``link_give_damage`` units."""
        return CONTACT_BY_MAIL[self.contact_class]

    def effect(self, weapon: str) -> int:
        """Give-damage byte for sword, arrow, bomb, fire, ice, hammer, or hookshot."""
        try:
            index = _WEAPON_CLASS[weapon]
        except KeyError:
            raise ValueError(f"unknown weapon {weapon!r}") from None
        return self.damage[index]

    def hurts(self, weapon: str) -> bool:
        """True when ``weapon`` subtracts HP or applies burn, freeze, blob, or fairy.

        Stun bytes and zero do not hurt. ``ignores_damage`` sprites never
        consult the table.
        """
        if self.ignores_damage:
            return False
        amount = self.effect(weapon)
        return amount != 0 and amount not in _STUN_EFFECTS

    @property
    def hookshot_stun(self) -> bool:
        """True when the hookshot byte stuns and the sprite can receive it."""
        if self.ignores_damage:
            return False
        return self.effect("hookshot") in _STUN_EFFECTS


_VULNERABILITY: tuple[SpriteVulnerability, ...] = _load_vulnerability()
VULNERABILITY: Mapping[int, SpriteVulnerability] = {
    index: row for index, row in enumerate(_VULNERABILITY)
}


def vulnerability(sprite_type: int) -> SpriteVulnerability:
    """Vulnerability row for a sprite type byte ``0x00``–``0xF2``."""
    if isinstance(sprite_type, bool):
        raise TypeError(f"sprite type must be an int, got {sprite_type!r}")
    try:
        index = operator.index(sprite_type)
    except TypeError:
        raise TypeError(f"sprite type must be an int, got {sprite_type!r}") from None
    if 0 <= index < len(_VULNERABILITY):
        return _VULNERABILITY[index]
    raise ValueError(f"sprite type {sprite_type!r} has no vulnerability row")
