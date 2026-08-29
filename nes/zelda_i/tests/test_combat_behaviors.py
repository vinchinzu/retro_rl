"""Unit tests for reusable Zelda I combat policies (no emulator)."""

from __future__ import annotations

from zelda_i.combat import FACING_SOUTH, SWORD_REACH
from zelda_i.dungeon.behaviors import (
    DIGDOGGER_POLICY,
    DIGDOGGER_SHRUNK_TYPE,
    DIGDOGGER_TYPE,
    FIREBALL_TYPE,
    GIBDO_TYPE,
    GORIYA_TYPE,
    KEESE_TYPE,
    POLS_VOICE_TYPE,
    ROPE_TYPE,
    STALFOS_TYPE,
    VIRE_SPLIT_KEESE_TYPE,
    WALLMASTER_TYPE,
    EnemyKind,
    blocked_by_projectile,
    default_alive_rule,
    engagement_hint,
    is_off_wall,
    is_shrunk,
    kind_for_type,
    live_among,
    liveness,
    needs_whistle,
    policy_for,
    projectile_threats,
    rope_on_axis,
    sword_legal,
    uses_type_only_liveness,
)
from zelda_i.dungeon.engine import AliveRule
from zelda_i.ram import ZeldaObject


def _obj(
    slot: int = 1,
    *,
    type_id: int = STALFOS_TYPE,
    x: int = 100,
    y: int = 100,
    hp: int = 0x20,
) -> ZeldaObject:
    return ZeldaObject(
        slot=slot,
        type_id=type_id,
        x=x,
        y=y,
        facing=FACING_SOUTH,
        hp=hp,
        state=0,
    )


def test_kind_for_type_catalog() -> None:
    assert kind_for_type(STALFOS_TYPE) is EnemyKind.STALFOS
    assert kind_for_type(KEESE_TYPE) is EnemyKind.KEESE
    assert kind_for_type(VIRE_SPLIT_KEESE_TYPE) is EnemyKind.KEESE
    assert kind_for_type(ROPE_TYPE) is EnemyKind.ROPE
    assert kind_for_type(GORIYA_TYPE) is EnemyKind.GORIYA
    assert kind_for_type(POLS_VOICE_TYPE) is EnemyKind.POLS_VOICE
    assert kind_for_type(GIBDO_TYPE) is EnemyKind.GIBDO
    assert kind_for_type(WALLMASTER_TYPE) is EnemyKind.WALLMASTER
    assert kind_for_type(DIGDOGGER_TYPE) is EnemyKind.DIGDOGGER
    assert kind_for_type(DIGDOGGER_SHRUNK_TYPE) is EnemyKind.DIGDOGGER
    assert kind_for_type(FIREBALL_TYPE) is EnemyKind.PROJECTILE
    assert kind_for_type(0x00) is EnemyKind.UNKNOWN


def test_keese_type_liveness_even_when_hp_is_zero() -> None:
    """Keese HP stays 0 while alive — the AGENTS.md trap."""
    keese = _obj(1, type_id=KEESE_TYPE, hp=0)
    split = _obj(2, type_id=VIRE_SPLIT_KEESE_TYPE, hp=0)
    empty = _obj(3, type_id=0, hp=0)
    dead_ff = _obj(4, type_id=0xFF, hp=0x20)

    assert uses_type_only_liveness(keese)
    assert uses_type_only_liveness(EnemyKind.KEESE)
    assert liveness(keese, AliveRule.TYPE)
    assert liveness(keese, AliveRule.TYPE_AND_HP)
    assert liveness(split, AliveRule.TYPE_AND_HP)
    assert not liveness(empty, AliveRule.TYPE)
    assert not liveness(dead_ff, AliveRule.TYPE_AND_HP)
    assert default_alive_rule(EnemyKind.KEESE) is AliveRule.TYPE

    live = live_among((keese, empty, dead_ff), AliveRule.TYPE_AND_HP)
    assert live == (keese,)


def test_stalfos_uses_hp_liveness_and_engage_distance() -> None:
    live = _obj(1, type_id=STALFOS_TYPE, x=150, y=141, hp=0x20)
    corpse = _obj(2, type_id=STALFOS_TYPE, x=150, y=141, hp=0)
    link = (120, 141)

    assert liveness(live, AliveRule.TYPE_AND_HP)
    assert liveness(live, AliveRule.TYPE)
    assert not liveness(corpse, AliveRule.TYPE_AND_HP)
    assert liveness(corpse, AliveRule.TYPE)
    assert default_alive_rule(EnemyKind.STALFOS) is AliveRule.TYPE_AND_HP

    far = engagement_hint(EnemyKind.STALFOS, link, live)
    assert far.preferred_distance == 48
    assert far.face == "RIGHT"
    assert not far.swing
    assert not far.retreat
    assert abs(live.x - link[0]) > SWORD_REACH

    close = _obj(1, type_id=STALFOS_TYPE, x=132, y=141, hp=0x20)
    hit = engagement_hint(EnemyKind.STALFOS, link, close)
    assert hit.face == "RIGHT"
    assert hit.swing
    assert not hit.retreat


def test_ropes_face_charge_lane() -> None:
    link = (120, 141)
    on_row = _obj(1, type_id=ROPE_TYPE, x=200, y=141, hp=0x20)
    on_col = _obj(2, type_id=ROPE_TYPE, x=120, y=80, hp=0x20)

    assert rope_on_axis(*link, on_row)
    assert rope_on_axis(*link, on_col)
    assert policy_for(EnemyKind.ROPE).preferred_distance == 64

    row_hint = engagement_hint(EnemyKind.ROPE, link, on_row)
    assert row_hint.preferred_distance == 64
    assert row_hint.face == "RIGHT"
    assert not row_hint.swing  # still outside the blade

    col_hint = engagement_hint(EnemyKind.ROPE, link, on_col)
    assert col_hint.face == "UP"

    in_blade = _obj(3, type_id=ROPE_TYPE, x=132, y=141, hp=0x20)
    slash = engagement_hint(EnemyKind.ROPE, link, in_blade)
    assert slash.face == "RIGHT"
    assert slash.swing


def test_projectile_threat_blocks_approach_not_rear() -> None:
    lx, ly = 128, 125
    fire = _obj(1, type_id=FIREBALL_TYPE, x=lx + 20, y=ly, hp=0)
    rear = _obj(2, type_id=FIREBALL_TYPE, x=lx - 40, y=ly, hp=0)
    stalfos = _obj(3, type_id=STALFOS_TYPE, x=lx + 20, y=ly, hp=0x20)

    assert projectile_threats(lx, ly, (fire,), direction="RIGHT") == (fire,)
    assert blocked_by_projectile(lx, ly, "RIGHT", (fire,))
    assert not blocked_by_projectile(lx, ly, "LEFT", (fire,))
    assert not blocked_by_projectile(lx, ly, "RIGHT", (rear,))
    assert not blocked_by_projectile(lx, ly, "RIGHT", (stalfos,))

    goriya = _obj(4, type_id=GORIYA_TYPE, x=lx + 40, y=ly, hp=0x30)
    threatened = engagement_hint(
        EnemyKind.GORIYA,
        (lx, ly),
        goriya,
        projectiles=(fire,),
    )
    assert threatened.retreat
    assert threatened.face == "RIGHT"

    clear = engagement_hint(
        EnemyKind.GORIYA,
        (lx, ly),
        goriya,
        projectiles=(),
    )
    assert not clear.retreat


def test_digdogger_whistle_gate() -> None:
    link = (120, 141)
    big = _obj(1, type_id=DIGDOGGER_TYPE, x=132, y=141, hp=240)
    small = _obj(2, type_id=DIGDOGGER_SHRUNK_TYPE, x=132, y=141, hp=128)

    assert needs_whistle(big)
    assert not is_shrunk(big)
    assert not sword_legal(EnemyKind.DIGDOGGER, big)
    assert "whistle" in DIGDOGGER_POLICY.lower()
    assert "0x38" in DIGDOGGER_POLICY

    big_hint = engagement_hint(EnemyKind.DIGDOGGER, link, big)
    assert not big_hint.swing
    assert big_hint.preferred_distance == 64

    assert is_shrunk(small)
    assert not needs_whistle(small)
    assert sword_legal(EnemyKind.DIGDOGGER, small)
    small_hint = engagement_hint(EnemyKind.DIGDOGGER, link, small)
    assert small_hint.swing


def test_wallmaster_only_after_leaving_wall() -> None:
    parked = _obj(1, type_id=WALLMASTER_TYPE, x=0, y=141, hp=0x20)
    in_room = _obj(2, type_id=WALLMASTER_TYPE, x=80, y=141, hp=0x20)
    link = (120, 141)

    assert not is_off_wall(parked)
    assert is_off_wall(in_room)
    assert not sword_legal(EnemyKind.WALLMASTER, parked)
    assert sword_legal(EnemyKind.WALLMASTER, in_room)

    parked_hint = engagement_hint(EnemyKind.WALLMASTER, link, parked)
    assert not parked_hint.swing
    assert parked_hint.preferred_distance == 80

    close = _obj(3, type_id=WALLMASTER_TYPE, x=108, y=141, hp=0x20)
    in_hint = engagement_hint(EnemyKind.WALLMASTER, link, close)
    assert in_hint.swing


def test_pols_and_gibdo_preferred_distance() -> None:
    assert policy_for(EnemyKind.POLS_VOICE).preferred_distance == 72
    assert policy_for(EnemyKind.GIBDO).preferred_distance == 56
    assert default_alive_rule(POLS_VOICE_TYPE) is AliveRule.TYPE_AND_HP
    assert default_alive_rule(GIBDO_TYPE) is AliveRule.TYPE_AND_HP
