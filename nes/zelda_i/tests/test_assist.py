"""Unit tests for survival assist (no emulator required)."""

from __future__ import annotations

from types import SimpleNamespace

from zelda_i.assist import UnlimitedHealthAssist, poke_wooden_arrows
from zelda_i.dungeon_ops import B_ITEM_ARROWS, WOODEN_ARROWS
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOW,
    ADDR_SELECTED_ITEM,
    PLAY_MODE,
    ZeldaSnapshot,
    full_health_byte,
)


def _snap(
    *,
    mode: int = PLAY_MODE,
    level: int = 0,
    health: int = 0x20,
    screen: int = 0x4A,
    link_x: int = 120,
    link_y: int = 141,
    room_item_id: int = 0,
) -> ZeldaSnapshot:
    return ZeldaSnapshot(
        mode=mode,
        level=level,
        screen=screen,
        next_screen=screen,
        link_x=link_x,
        link_y=link_y,
        facing=8,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=health,
        triforce=1,
        compass=0,
        dialog_timer=0,
        colliding_tile=0x26,
        room_item_id=room_item_id,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=(),
    )


class _FakeData:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def set_value(self, key: str, value: int) -> None:
        self.values[key] = int(value)


def test_full_health_byte_preserves_containers() -> None:
    assert full_health_byte(0x20) == 0x22
    assert full_health_byte(0x31) == 0x33
    assert full_health_byte(0x2F) == 0x22


def test_assist_refills_on_ordinary_play() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x20), frame=10)
    assert data.values["health"] == 0x22
    assert data.values["heart_partial"] == 0xFF
    assert assist.telemetry.health.writes == 1
    assert assist.telemetry.health.restored == 2
    assert assist.telemetry.health.first_active_frame == 10
    assert assist.telemetry.progression_writes == 0
    assert assist.telemetry.capacity_writes == 0


def test_assist_skips_when_full() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x22), frame=1)
    assert data.values == {}
    assert assist.telemetry.health.writes == 0


def test_assist_clamps_transient_container_jump() -> None:
    """A mid-play high-nibble spike must not lock extra hearts."""
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x22), frame=1)
    assert assist.telemetry.accepted_containers == 3

    data.values.clear()
    # Tape bug: fill INC from a 0xF write, or a transient 0x6F (3 → 7).
    assist.apply_snapshot(data, _snap(health=0x6F), frame=2)
    assert data.values["health"] == 0x22
    assert assist.telemetry.accepted_containers == 3
    assert assist.telemetry.container_clamps >= 1
    assert assist.telemetry.capacity_writes == 0


def test_assist_accepts_heart_container_plus_one() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x22), frame=1)
    data.values.clear()
    assist.apply_snapshot(
        data, _snap(health=0x33, room_item_id=0x1A), frame=2
    )
    assert assist.telemetry.accepted_containers == 4
    assert data.values == {}
    assert assist.telemetry.container_clamps == 0


class _AssignMem:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, int]] = []

    def assign(self, addr: int, fmt: str, val: int) -> None:
        self.calls.append((addr, fmt, val))


def _env_with_mem(mem: object) -> SimpleNamespace:
    data = SimpleNamespace(memory=mem)
    return SimpleNamespace(unwrapped=SimpleNamespace(data=data))


def test_poke_wooden_arrows_writes_arrows_and_b_not_bow() -> None:
    mem = _AssignMem()
    report = poke_wooden_arrows(_env_with_mem(mem), from_arrows=0, select=True)
    assert mem.calls == [
        (ADDR_ARROWS, "|u1", WOODEN_ARROWS),
        (ADDR_SELECTED_ITEM, "|u1", B_ITEM_ARROWS),
    ]
    addrs = [addr for addr, _fmt, _val in mem.calls]
    assert ADDR_BOW not in addrs
    assert report["inventory_writes"] == 1
    assert report["poke_arrows"] == WOODEN_ARROWS
    assert report["progression_writes"] == 0
    assert report["capacity_writes"] == 0
    assert report["bow_writes"] == 0
    assert report["state_load"] is False


def test_poke_wooden_arrows_skips_count_when_already_wooden() -> None:
    mem = _AssignMem()
    report = poke_wooden_arrows(_env_with_mem(mem), from_arrows=1, select=True)
    assert mem.calls == [(ADDR_SELECTED_ITEM, "|u1", B_ITEM_ARROWS)]
    assert report["inventory_writes"] == 0
    assert report["poke_arrows"] == 0
    assert report["progression_writes"] == 0
