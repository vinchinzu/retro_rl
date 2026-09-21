"""ROM constants per object type, and the seams they must not duplicate.

No emulator. Every number here is transcribed from
``aldonunez/zelda1-disassembly`` at commit ``50a1c86`` and sourced line by
line in ``scratch/enemy_constants_rom.md``; a test that disagrees with the ROM
is the thing to fix, not the disassembly (see ``test_respawn.py``).

Two kinds of test live here and they are not the same claim:

* **Structural** -- lookup totality, frozen-ness, unique ids, every constant
  carrying a source URL. These fail on a refactor.
* **Behavioural** -- a sourced ROM number, and the agreement between this
  table and the two places ``dungeon/behaviors.py::KIND_POLICY`` encodes the
  same fact as *policy*. These fail when someone changes what the tree
  believes about the game.
"""

from __future__ import annotations

import dataclasses

import pytest

from zelda_i import combat
from zelda_i.dungeon import behaviors, threat
from zelda_i.dungeon.ids import AliveRule
from zelda_i.dungeon.species import (
    ARROW_DAMAGE,
    BURROWER_DORMANT_STATE,
    CONTACT_THRESHOLD,
    DEFAULT_BODY_HALF,
    DEFAULT_QSPEED,
    DEFAULT_SHOT_HALF,
    DMG_ARROW,
    DMG_SWORD,
    HALF_WIDTH_MID_OFFSET,
    LINK_QSPEED,
    LINK_STAIRS_QSPEED,
    MID_OFFSET,
    QSPEED_PER_PIXEL,
    SWORD_DAMAGE,
    Contact,
    Species,
    _TABLE,
    species_of,
)

_PINNED_SOURCE = (
    "https://github.com/aldonunez/zelda1-disassembly/blob/"
    "50a1c869a8d8e2eb8b5b60acea325f44b4341762/src/"
)


# --- Structural ------------------------------------------------------------


def test_every_byte_id_resolves_and_never_raises() -> None:
    """Structural: the accessor is total over a RAM byte."""
    for type_id in range(0x100):
        row = species_of(type_id)
        assert row.type_id == type_id
        assert row.mid_offset_y == MID_OFFSET


def test_unknown_id_falls_back_to_todays_threat_constants() -> None:
    """Structural: an unlisted id keeps the behaviour `threat.py` has today."""
    row = species_of(0xAB)
    assert row.hp is None
    assert row.damage_points is None
    assert row.contact is Contact.HARM
    assert row.attr == 0
    assert row.mid_offset_x == DEFAULT_BODY_HALF
    assert row.speed is None
    assert row.qspeed() == DEFAULT_QSPEED
    assert row.invincible_to == 0


def test_fallback_halves_still_equal_threat() -> None:
    """Behavioural: the tie that makes 'nothing changes' true.

    `species` deliberately does not import `threat`; this is the assertion
    that keeps the two copies of 8 / 4 honest.
    """
    assert DEFAULT_BODY_HALF == threat.BODY_HALF
    assert DEFAULT_SHOT_HALF == threat.SHOT_HALF


def test_rows_are_frozen() -> None:
    row = species_of(0x07)
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.hp = 99  # type: ignore[misc]


def test_no_duplicate_or_mismatched_ids() -> None:
    assert len(_TABLE) == len({row.type_id for row in _TABLE.values()})
    for key, row in _TABLE.items():
        assert key == row.type_id
        assert isinstance(row, Species)
    assert 0 not in _TABLE, "type 0 is the empty slot, not a species"


def test_every_sourced_number_carries_a_url() -> None:
    """Structural: a constant with no URL is not allowed in the table."""
    unsourced = [
        f"0x{row.type_id:02X}"
        for row in _TABLE.values()
        if row.speed is not None and not row.speed.url.startswith(_PINNED_SOURCE)
    ]
    assert not unsourced, "speed without a pinned source URL: " + ", ".join(unsourced)
    assert any(row.speed is not None for row in _TABLE.values())


# --- Behavioural: hitbox ---------------------------------------------------


def test_contact_threshold_is_the_rom_nine() -> None:
    """Z_01.asm#L5647: one 9 px centre distance, bodies and shots alike."""
    assert CONTACT_THRESHOLD == 9


@pytest.mark.parametrize(
    "type_id",
    [0x14, 0x15, 0x25, 0x33, 0x34, 0x43, 0x53, 0x55, 0x5B, 0x5C],
)
def test_half_width_types_move_the_x_midpoint(type_id: int) -> None:
    """Attribute $40 (Z_01.asm#L5553) is the only per-type hitbox datum."""
    row = species_of(type_id)
    assert row.half_width
    assert row.mid_offset_x == HALF_WIDTH_MID_OFFSET
    assert row.mid_offset_y == MID_OFFSET


@pytest.mark.parametrize("type_id", [0x07, 0x0B, 0x11, 0x1B, 0x2A, 0x30])
def test_full_width_types_keep_todays_eight(type_id: int) -> None:
    row = species_of(type_id)
    assert not row.half_width
    assert row.mid_offset_x == threat.BODY_HALF


# --- Behavioural: HP -------------------------------------------------------


@pytest.mark.parametrize(
    ("type_id", "hp"),
    [
        (0x01, 96),  # blue Lynel
        (0x02, 64),  # red Lynel
        (0x03, 48),  # blue Moblin
        (0x04, 32),  # red Moblin
        (0x07, 16),  # red Octorok
        (0x09, 32),  # blue Octorok
        (0x0B, 64),  # Darknut
        (0x0E, 16),  # Tektite
        (0x0F, 64),  # blue Leever
        (0x10, 32),  # red Leever
        (0x11, 32),  # Zora
        (0x12, 64),  # Vire -- live rr-5lu said HP 64
        (0x13, 32),  # Zol
        (0x15, 0),  # Gel
        (0x16, 160),  # Pol's Voice
        (0x17, 144),  # Like-Like
        (0x18, 128),  # shrunk Digdogger
        (0x1A, 32),  # Peahat
        (0x1B, 0),  # Keese -- HP stays 0 while alive
        (0x1C, 0),  # Vire-split Keese
        (0x1D, 0),  # black Keese
        (0x1E, 48),  # Armos
        (0x21, 144),  # Ghini
        (0x23, 160),  # blue Wizzrobe
        (0x24, 64),  # orange Wizzrobe
        (0x27, 32),  # Wallmaster
        (0x28, 16),  # Rope
        (0x2A, 32),  # Stalfos
        (0x2B, 240),  # the L3 invulnerable mover
        (0x30, 112),  # Gibdo
        (0x32, 240),  # Dodongo
        (0x33, 96),  # Gohma, type $33
        (0x34, 32),  # Gohma, type $34
        (0x38, 240),  # Digdogger, large
        (0x3C, 64),  # Manhandla segment
        (0x3D, 96),  # Aquamentus
        (0x43, 160),  # Gleeok
        (0x47, 176),  # Patra
        (0x40, 240),  # Bubble
    ],
)
def test_hp_matches_rom(type_id: int, hp: int) -> None:
    """ObjectTypeToHpPairs (Z_07.asm#L5256) via ExtractHitPointValue."""
    assert species_of(type_id).hp == hp


def test_shots_have_no_hp_entry() -> None:
    """ObjectTypeToHpPairs stops at type $4B; shots are blank, not zero."""
    for type_id in (0x53, 0x55, 0x57, 0x5B, 0x5C):
        assert species_of(type_id).hp is None


# --- Behavioural: damage ---------------------------------------------------


def test_sword_damage_points_are_16_32_64() -> None:
    """SwordDamagePoints (Z_01.asm#L6162)."""
    assert SWORD_DAMAGE == (0x10, 0x20, 0x40)
    assert ARROW_DAMAGE == (0x20, 0x40)


def test_hits_to_kill_from_hp_and_damage() -> None:
    """DealDamage kills when damage points >= HP (Z_01.asm#L5969)."""
    aquamentus = species_of(0x3D)
    assert aquamentus.hits_to_kill(SWORD_DAMAGE[0]) == 6
    assert aquamentus.hits_to_kill(SWORD_DAMAGE[1]) == 3
    assert aquamentus.hits_to_kill(SWORD_DAMAGE[2]) == 2
    assert species_of(0x1B).hits_to_kill(SWORD_DAMAGE[0]) == 1
    assert species_of(0x33).hits_to_kill(ARROW_DAMAGE[0]) == 3
    assert species_of(0x34).hits_to_kill(ARROW_DAMAGE[0]) == 1
    assert species_of(0x53).hits_to_kill(SWORD_DAMAGE[0]) is None


def test_rock_costs_exactly_half_a_heart() -> None:
    """ObjTypeToDamagePoints[$53] = $80 -- the live $FF -> $7F partial."""
    rock = species_of(0x53)
    assert rock.damage_points == 0x80
    assert rock.contact_hearts == pytest.approx(0.5)


def test_bubbles_cost_no_hearts_and_are_still_contact() -> None:
    """$2B-$2E are $00: HarmLink runs, hearts do not move, the streak does."""
    for type_id in (0x2B, 0x2C, 0x2D, 0x2E):
        row = species_of(type_id)
        assert row.damage_points == 0
        assert row.contact_hearts == 0.0
        assert row.contact is Contact.HARMLESS


def test_capture_types_are_not_a_heart_cost() -> None:
    """Wallmaster / Like-Like take Link (Z_01.asm#L5528), a different fact."""
    assert species_of(0x27).contact is Contact.CAPTURE
    assert species_of(0x17).contact is Contact.CAPTURE
    assert species_of(0x02).contact is Contact.HARM


@pytest.mark.parametrize(
    ("type_id", "hearts"),
    [(0x01, 2.0), (0x02, 1.0), (0x04, 0.5), (0x0B, 1.0), (0x30, 2.0)],
)
def test_contact_damage_matches_rom(type_id: int, hearts: float) -> None:
    assert species_of(type_id).contact_hearts == pytest.approx(hearts)


def test_invincibility_masks_gate_the_weapon() -> None:
    """ObjInvincibilityMask & damage type != 0 means immune (Z_01.asm#L5921)."""
    gohma = species_of(0x33)
    assert gohma.weak_to(DMG_ARROW)
    assert not gohma.weak_to(DMG_SWORD)
    darknut = species_of(0x0B)
    assert darknut.weak_to(DMG_SWORD)
    assert not darknut.weak_to(DMG_ARROW)


def test_attribute_20_types_take_no_weapon_at_all() -> None:
    """Digdogger $38 and the blade trap $49 skip the whole weapon pass."""
    for type_id in (0x38, 0x49):
        row = species_of(type_id)
        assert row.weapon_proof
        assert not row.weak_to(DMG_SWORD)
    assert not species_of(0x18).weapon_proof  # shrunk Digdogger is sword-legal


# --- Behavioural: speed ----------------------------------------------------


def test_qspeed_unit_is_sixty_four() -> None:
    """MoveObject applies the quarter speed 4x a frame (Z_07.asm#L2768)."""
    assert QSPEED_PER_PIXEL == 64
    assert DEFAULT_QSPEED / QSPEED_PER_PIXEL == 0.5


def test_link_walks_one_and_a_half_pixels_a_frame() -> None:
    """Z_05.asm#L7123. `threat.LINK_SPEED` is 1.0; the ROM says 1.5."""
    assert LINK_QSPEED / QSPEED_PER_PIXEL == 1.5
    assert LINK_STAIRS_QSPEED / QSPEED_PER_PIXEL == 0.75


@pytest.mark.parametrize(
    ("type_id", "px_per_frame"),
    [
        (0x07, 0.5),  # slow Octorok
        (0x08, 1.0),  # fast Octorok
        (0x0B, 0.5),  # red Darknut
        (0x0C, 0.625),  # blue Darknut
        (0x27, 0.375),  # Wallmaster
        (0x2A, 0.5),  # Stalfos
        (0x40, 1.0),  # Bubble
        (0x55, 1.75),  # fireball / Zora spit
        (0x5B, 2.0),  # Moblin arrow
        (0x5C, 2.5),  # Goriya boomerang
    ],
)
def test_speed_matches_rom(type_id: int, px_per_frame: float) -> None:
    assert species_of(type_id).px_per_frame() == pytest.approx(px_per_frame)


def test_red_leever_only_moves_in_state_three() -> None:
    """RedLeeverStateQSpeeds (Z_04.asm#L2735) -- the ROM behind dormant_body."""
    leever = species_of(0x10)
    assert [leever.px_per_frame(state) for state in range(6)] == [
        0.0,
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
    ]


def test_blue_leever_creeps_even_while_buried() -> None:
    """BlueLeeverStateQSpeeds state 0 is $08 -- immobile is a red-leever fact."""
    assert species_of(0x0F).px_per_frame(0) == pytest.approx(0.125)


def test_types_without_a_sourced_speed_report_the_room_default() -> None:
    """No `STA ObjQSpeedFrac` of their own: they run at InitMode_EnterRoom's."""
    for type_id in (0x16, 0x17, 0x1E, 0x23, 0x30):
        row = species_of(type_id)
        assert row.speed is None
        assert row.qspeed() == DEFAULT_QSPEED


# --- Live measurement vs the ROM -------------------------------------------


def test_zora_shot_speed_measured_live_equals_the_rom() -> None:
    """`behaviors.ZORA_SHOT_SPEED` came off four live surfacings; the ROM's
    FireballQSpeedsX[0] is $70 (Z_04.asm#L981). They agree exactly."""
    assert behaviors.ZORA_SHOT_SPEED == pytest.approx(species_of(0x55).px_per_frame())


def test_sword_half_width_matches_the_rom_cross_axis() -> None:
    """`combat.SWORD_HALF_WIDTH` was tuned; the ROM's stab threshold is $0C."""
    assert combat.SWORD_HALF_WIDTH == 0x0C


# --- Reconciliation with KIND_POLICY (fact vs policy) ----------------------


def test_type_only_kinds_have_zero_rom_hp() -> None:
    """`alive_rule` stays policy; ROM HP is the reason it is TYPE, not a copy."""
    type_only = {
        kind for kind, policy in behaviors.KIND_POLICY.items() if policy.type_only
    }
    assert type_only, "expected at least Keese to be type-only"
    for type_id, kind in behaviors._TYPE_TO_KIND.items():
        if kind in type_only:
            assert species_of(type_id).hp == 0, f"0x{type_id:02X} is not HP-0 in ROM"


def test_rom_hp_zero_types_never_get_an_hp_liveness_rule() -> None:
    """A mapped type with ROM HP 0 must not be judged alive by hp > 0."""
    offenders = []
    for type_id, kind in behaviors._TYPE_TO_KIND.items():
        row = species_of(type_id)
        if row.hp != 0:
            continue
        policy = behaviors.KIND_POLICY[kind]
        if policy.alive_rule is not AliveRule.TYPE and not policy.type_only:
            offenders.append(f"0x{type_id:02X}->{kind.value}")
    assert not offenders, "HP-0 type judged by hp>0: " + ", ".join(offenders)


def test_whistle_then_sword_is_the_invincible_attribute() -> None:
    """DIGDOGGER's policy flag is attribute $20 on $38, cleared on $18."""
    whistle_kinds = {
        kind
        for kind, policy in behaviors.KIND_POLICY.items()
        if policy.whistle_then_sword
    }
    assert whistle_kinds
    mapped = {
        type_id
        for type_id, kind in behaviors._TYPE_TO_KIND.items()
        if kind in whistle_kinds
    }
    assert any(species_of(type_id).weapon_proof for type_id in mapped)
    assert any(not species_of(type_id).weapon_proof for type_id in mapped)


def test_dormant_body_types_are_the_rom_burrowers() -> None:
    """`combat.dormant_body` is Burrower state 0 (Z_04.asm#L2664)."""
    for type_id in combat.LEEVER_TYPES:
        row = species_of(type_id)
        assert row.no_contact_states == (BURROWER_DORMANT_STATE,)
        assert row.contact_in_state(BURROWER_DORMANT_STATE) is Contact.NONE
        assert row.contact_in_state(2) is Contact.HARM
    assert combat.LEEVER_DORMANT_STATE == BURROWER_DORMANT_STATE


def test_zora_shares_the_burrower_dormant_rule() -> None:
    """UpdateZora runs the same updater, so a submerged Zora is dormant too.

    `combat.no_contact_frame` now exposes it (rr-ttyu.4). `combat.dormant_body`
    still does **not**, on purpose: the measured `behaviors.ZORA_CYCLE` puts
    ObjState 0 at two frames of *surfacing*, not the 96-frame submerged phase
    (ObjState 5), and the Zora is answered by a clock rung that counts from
    state 0 rather than as a body. See `tests/test_enemy_motion_seam.py`.
    """
    zora = species_of(0x11)
    assert zora.no_contact_states == (BURROWER_DORMANT_STATE,)
    assert zora.contact_in_state(BURROWER_DORMANT_STATE) is Contact.NONE
    assert 0x11 not in combat.LEEVER_TYPES
    assert 0x11 in combat.DORMANT_BURROWER_TYPES


def test_species_does_not_duplicate_kind_policy_fields() -> None:
    """Fact and policy stay separate: no engagement field leaks into a row."""
    field_names = {field.name for field in dataclasses.fields(Species)}
    # `notes` is free prose on both sides, not a fact stored twice.
    policy_names = {
        field.name
        for field in dataclasses.fields(behaviors.KindPolicy)
        if field.name != "notes"
    }
    assert not (field_names & policy_names), (
        "species row duplicates KIND_POLICY fields: "
        + ", ".join(sorted(field_names & policy_names))
    )
