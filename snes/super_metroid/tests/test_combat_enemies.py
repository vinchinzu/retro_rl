"""Room-enemy Overlay: scan + Stance. ROM-free RAM / Enemy fixtures."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import super_metroid.combat.enemies as enemy_overlay
from super_metroid.combat.enemies import (
    ATOMIC_ID,
    COVERN_ID,
    WORKROBOT_ID,
    Enemy,
    Intent,
    Stance,
    choose,
    list_enemies,
)
from super_metroid.ram import FACING_LEFT, FACING_RIGHT
from super_metroid.routes.kpdr.k6.ws_basement_ice import (
    BASEMENT_ICE,
    ice_keepaway_action as basement_ice_keepaway,
    workrobot_avoid_action,
)
from super_metroid.routes.kpdr.k6.ws_main_ice import (
    SHAFT_ICE,
    ice_keepaway_action as shaft_ice_keepaway,
)
from super_metroid.routes.skills.charge_shot import CHARGE_FULL


def _enemy(
    enemy_id: int,
    x: int,
    y: int,
    *,
    hp: int = 250,
    freeze: int = 0,
    slot: int = 0,
) -> Enemy:
    return Enemy(slot, enemy_id, x, y, hp, freeze)


def test_list_enemies_reads_freeze_and_drops_dead() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = 0x0F78
    ram[base] = ATOMIC_ID & 0xFF
    ram[base + 1] = ATOMIC_ID >> 8
    ram[base + 0x02] = 1150 & 0xFF
    ram[base + 0x03] = 1150 >> 8
    ram[base + 0x06] = 1163 & 0xFF
    ram[base + 0x07] = 1163 >> 8
    ram[base + 0x14] = 250
    ram[base + 0x26] = 40
    dead = 0x0F78 + 0x40
    ram[dead] = ATOMIC_ID & 0xFF
    ram[dead + 1] = ATOMIC_ID >> 8
    ram[dead + 0x14] = 0

    found = list_enemies(ram)
    assert len(found) == 1
    assert found[0].enemy_id == ATOMIC_ID
    assert found[0].x == 1150
    assert found[0].freeze_timer == 40


def test_list_enemies_empty_without_ram() -> None:
    assert list_enemies(None) == ()
    assert list_enemies(object()) == ()


def test_list_enemies_drops_off_map() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = 0x0F78
    ram[base] = ATOMIC_ID & 0xFF
    ram[base + 1] = ATOMIC_ID >> 8
    ram[base + 0x02] = 0x00
    ram[base + 0x03] = 0xFE  # x = 0xFE00
    ram[base + 0x14] = 250
    assert list_enemies(ram) == ()


def test_list_enemies_scans_all_slots_once_and_keeps_unknown_ids() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = 0x0F78 + 31 * 0x40
    ram[base] = 0xEF
    ram[base + 1] = 0xBE
    ram[base + 0x02] = 100
    ram[base + 0x06] = 120
    ram[base + 0x14] = 1

    class _Env:
        reads = 0

        def get_ram(self):
            self.reads += 1
            return ram

    env = _Env()
    found = list_enemies(SimpleNamespace(env=env))
    assert env.reads == 1
    assert found == (_enemy(0xBEEF, 100, 120, hp=1, slot=31),)


def test_unknown_species_is_ignore() -> None:
    blob = _enemy(0xBEEF, 100, 100)
    choice = choose(100, 100, FACING_LEFT, (blob,), SHAFT_ICE)
    assert choice.buttons is None
    assert choice.stance is Stance.IGNORE


def test_covern_absorb_vs_engage() -> None:
    covern = _enemy(COVERN_ID, 1129, 1818, hp=80)
    absorb = choose(1075, 1845, FACING_LEFT, (covern,), BASEMENT_ICE)
    assert absorb.stance is Stance.ABSORB
    assert absorb.target is covern
    assert absorb.buttons is None
    ice = choose(1075, 1845, FACING_LEFT, (covern,), SHAFT_ICE)
    assert ice.buttons is not None
    assert ice.stance is Stance.ENGAGE


def test_shaft_ice_skips_pit_and_waits_on_frozen() -> None:
    blob = _enemy(ATOMIC_ID, 1150, 1160)
    pit = shaft_ice_keepaway(1173, 1979, FACING_LEFT, (blob,))
    assert pit is None
    shot = shaft_ice_keepaway(1152, 1163, FACING_LEFT, (blob,))
    assert shot is not None
    assert "X" in shot or "A" in shot
    none = shaft_ice_keepaway(1152, 1163, FACING_LEFT, ())
    assert none is None
    frozen = _enemy(ATOMIC_ID, 1152, 1163, freeze=80)
    wait = shaft_ice_keepaway(1152, 1163, FACING_LEFT, (frozen,))
    assert wait == ()
    nearly_thawed = frozen._replace(freeze_timer=1)
    assert shaft_ice_keepaway(1152, 1163, FACING_LEFT, (nearly_thawed,)) == ()
    engaged = choose(1152, 1163, FACING_LEFT, (blob,), SHAFT_ICE)
    assert engaged.stance is Stance.ENGAGE
    assert engaged.buttons is not None


def test_basement_ice_until_dead_and_robot_clamp() -> None:
    blob = _enemy(ATOMIC_ID, 638, 168)
    shot = basement_ice_keepaway(670, 185, FACING_LEFT, (blob,))
    assert shot is not None
    assert "X" in shot
    assert "LEFT" not in shot
    face = basement_ice_keepaway(670, 185, FACING_RIGHT, (blob,))
    assert face == ("LEFT",)
    frozen = basement_ice_keepaway(
        670, 185, FACING_LEFT, (_enemy(ATOMIC_ID, 638, 168, freeze=180),)
    )
    assert frozen is not None
    assert "X" in frozen
    blocked = basement_ice_keepaway(879, 187, FACING_LEFT, (blob,))
    assert blocked is not None
    assert blocked[0] == "LEFT"
    assert "B" in blocked
    map_side = basement_ice_keepaway(
        900, 185, FACING_LEFT, (_enemy(ATOMIC_ID, 152, 77),)
    )
    assert map_side is None
    dead = basement_ice_keepaway(
        670, 185, FACING_LEFT, (_enemy(ATOMIC_ID, 638, 168, hp=0),)
    )
    assert dead is None
    overlap = basement_ice_keepaway(638, 168, FACING_LEFT, (blob,))
    assert overlap == ("X",)
    robot = _enemy(WORKROBOT_ID, 624, 176, hp=800, slot=1)
    release = basement_ice_keepaway(
        672,
        187,
        FACING_LEFT,
        (blob, robot),
        charge=CHARGE_FULL,
        velocity_y=2,
    )
    assert release is not None
    assert "X" not in release
    assert "A" in release
    turning = basement_ice_keepaway(672, 187, FACING_LEFT, (blob,), movement_type=14)
    assert turning == ("LEFT",)
    assert "X" not in turning


def test_workrobot_avoid_and_stall() -> None:
    robot = _enemy(WORKROBOT_ID, 880, 176, hp=800)
    wait = workrobot_avoid_action(900, 185, (robot,))
    assert wait == ()
    flee = workrobot_avoid_action(
        657, 187, (_enemy(WORKROBOT_ID, 657, 176, hp=800),)
    )
    assert flee == ("RIGHT", "B")
    clear = workrobot_avoid_action(
        740, 185, (_enemy(WORKROBOT_ID, 657, 176, hp=800),)
    )
    assert clear is None


def test_atomic_default_is_engage() -> None:
    blob = _enemy(ATOMIC_ID, 100, 100)
    assert choose(100, 100, FACING_LEFT, (blob,), Intent()).stance is Stance.ENGAGE


def test_intent_rejects_conflicting_stance_overrides() -> None:
    with pytest.raises(ValueError, match="multiple Stances.*0xE9FF"):
        Intent(
            engage=frozenset({ATOMIC_ID}),
            ignore=frozenset({ATOMIC_ID}),
        )


def test_overlay_has_only_two_public_functions() -> None:
    public_functions = {
        name
        for name in enemy_overlay.__all__
        if inspect.isfunction(getattr(enemy_overlay, name))
    }
    assert public_functions == {"choose", "list_enemies"}
