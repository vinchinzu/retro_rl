"""Unit tests for survival assist (no emulator required)."""

from __future__ import annotations

from types import SimpleNamespace

from zelda_i.assist import LastHeartAssist, UnlimitedHealthAssist, poke_wooden_arrows
from zelda_i.dungeon.ops import B_ITEM_ARROWS, WOODEN_ARROWS
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOW,
    ADDR_SELECTED_ITEM,
    PLAY_MODE,
    ZeldaSnapshot,
    full_health_byte,
    health_byte_is_coherent,
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


def test_health_byte_coherence_rejects_more_hearts_than_containers() -> None:
    """The shape ``Level6Entrance`` broke: lo=15 whole hearts in hi=3 slots."""
    assert health_byte_is_coherent(0x22) is True
    assert health_byte_is_coherent(0x66) is True
    assert health_byte_is_coherent(0x32) is True
    assert health_byte_is_coherent(0x20) is True
    assert health_byte_is_coherent(0x2F) is False
    assert health_byte_is_coherent(0x23) is False


def test_full_health_byte_output_is_always_coherent() -> None:
    for hi in range(16):
        for lo in range(16):
            assert health_byte_is_coherent(full_health_byte(hi << 4 | lo))


def test_last_heart_leaves_two_hearts_and_refills_the_last() -> None:
    """Two hearts stay real. The last heart is the only write, and it is full."""
    data = _FakeData()
    assist = LastHeartAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x22), frame=1)
    assert data.values == {}
    assist.apply_snapshot(data, _snap(health=0x21), frame=2)
    assert data.values == {}
    assert assist.telemetry.health.writes == 0
    assist.apply_snapshot(data, _snap(health=0x20), frame=3)
    assert data.values["health"] == 0x22
    assert data.values["heart_partial"] == 0xFF
    assert assist.telemetry.health.writes == 1
    # 3→2 and 2→1 are both real damage. Only the second one writes.
    assert assist.telemetry.total_damage == 2
    assert assist.telemetry.capacity_writes == 0
    assert assist.telemetry.progression_writes == 0
    report = assist.report()
    assert report["kind"] == "last_heart"
    assert report["engage_at_whole_hearts"] == 1


def test_observed_damage_guard_refills_after_a_survived_two_heart_hit() -> None:
    data = _FakeData()
    assist = LastHeartAssist(enabled=True, observed_damage_guard=True)
    assist.apply_snapshot(data, _snap(health=0x43), frame=1)  # 4 of 5 hearts
    assert data.values == {}
    assist.apply_snapshot(data, _snap(health=0x41), frame=2)  # 2-heart hit
    assert data.values["health"] == 0x44
    report = assist.report()
    assert report["maximum_single_frame_damage"] == 2
    assert report["effective_floor"] == 2
    assert report["safety_refills"] == 1
    assert report["target_refills"] == 0
    assert report["kind"] == "guarded_last_heart"


def test_observed_damage_guard_keeps_one_heart_refills_separate() -> None:
    data = _FakeData()
    assist = LastHeartAssist(enabled=True, observed_damage_guard=True)
    assist.apply_snapshot(data, _snap(health=0x41), frame=1)
    assert data.values == {}
    assist.apply_snapshot(data, _snap(health=0x40), frame=2)
    assert data.values["health"] == 0x44
    report = assist.report()
    assert report["safety_refills"] == 0
    assert report["target_refills"] == 1


def test_two_heart_threshold_reports_its_actual_floor() -> None:
    data = _FakeData()
    assist = UnlimitedHealthAssist(enabled=True, engage_at_whole_hearts=2)
    assist.apply_snapshot(data, _snap(health=0x43), frame=1)
    assist.apply_snapshot(data, _snap(health=0x41), frame=2)
    assert data.values["health"] == 0x44
    assert assist.report()["kind"] == "threshold_health"
    assert assist.report()["effective_floor"] == 2


def test_last_heart_does_not_grant_a_container() -> None:
    data = _FakeData()
    assist = LastHeartAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x22), frame=1)
    assist.apply_snapshot(data, _snap(health=0x6F), frame=2)
    assert data.values == {}
    assert assist.telemetry.accepted_containers == 3
    assert assist.telemetry.container_clamps >= 1
    assert assist.telemetry.capacity_writes == 0


def test_last_heart_refill_uses_the_owned_container_count() -> None:
    data = _FakeData()
    assist = LastHeartAssist(enabled=True)
    assist.apply_snapshot(data, _snap(health=0x30), frame=1)
    assert data.values["health"] == 0x33
    assert assist.telemetry.accepted_containers == 4


def test_unlimited_report_kind_stays_unlimited() -> None:
    assist = UnlimitedHealthAssist(enabled=True)
    assert assist.report()["kind"] == "unlimited_health"
    assert assist.report()["engage_at_whole_hearts"] is None


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
