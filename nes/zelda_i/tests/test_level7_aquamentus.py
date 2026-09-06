"""L7 0x2A Aquamentus kill + heart container (`level7.aquamentus`, live 2/2).

Live evidence (rr-8t4.3, 2026-09-04): `20260904_W3`-`W6` on the walk-on
lineage (cleared 0x0D pin -> stairs -> cellar 0x7B -> 0x29 -> bomb-E), HC
3 -> 4 every run, 398 controller frames, `position_writes=0`. The pickup cell
`(136,141)` comes from `scratch/probe_l7_2a_heart.py` (`20260904_H1`), which
dumped all 13 object slots after the kill: the container is not an object,
and L1's fixed `(192,141)` cell does not collect it.
"""

from __future__ import annotations

from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from zelda_i.level7.aquamentus import (
    HEART_CELL,
    HEART_STALL_FRAMES,
    HEART_SWEEP_MAX_FRAMES,
    HEART_SWEEP_WAYPOINTS,
    Level7AquamentusHeartController,
    make_level7_aquamentus_heart_controller,
)
from zelda_i.level7.hops import make_aquamentus_heart_controller
from zelda_i.level7.stairs import AQUAMENTUS_ROM
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 7,
    "screen": AQUAMENTUS_ROM,
}


def _snap(
    *,
    x: int,
    y: int,
    containers: int = 3,
    screen: int = AQUAMENTUS_ROM,
    mode: int = PLAY_MODE,
    level: int = 7,
):
    return read_snapshot(
        make_ram(
            _DEFAULTS,
            x=x,
            y=y,
            screen=screen,
            mode=mode,
            level=level,
            health=((containers - 1) << 4) | (containers - 1),
        )
    )


def _buttons(action) -> list[str]:
    return sorted(
        name
        for name, idx in NES_BUTTON_NAME_TO_INDEX.items()
        if idx is not None and int(action.action[idx])
    )


def _after_kill(**kwargs) -> Level7AquamentusHeartController:
    return Level7AquamentusHeartController(boss_defeated=True, **kwargs)


def test_first_sweep_waypoint_is_the_measured_pickup_cell() -> None:
    assert HEART_CELL == (136, 141)
    assert HEART_SWEEP_WAYPOINTS[0] == HEART_CELL


def test_hops_factory_returns_the_live_controller() -> None:
    controller = make_aquamentus_heart_controller()
    assert isinstance(controller, Level7AquamentusHeartController)
    assert controller.report()["route_eligible"] is False


def test_sweep_walks_toward_the_pickup_cell_after_the_kill() -> None:
    controller = _after_kill()
    action = controller.step(_snap(x=190, y=125))
    assert _buttons(action) == ["DOWN"]
    action = controller.step(_snap(x=190, y=141))
    assert _buttons(action) == ["LEFT"]


def test_success_is_the_rising_edge_of_the_container_count() -> None:
    controller = _after_kill()
    controller.step(_snap(x=190, y=141, containers=3))
    assert not controller.success
    controller.step(_snap(x=160, y=141, containers=4))
    assert controller.success
    assert "heart_container_collected" in controller.notes


def test_a_pin_that_already_has_four_containers_never_greens() -> None:
    controller = _after_kill()
    for _ in range(50):
        controller.step(_snap(x=190, y=141, containers=4))
    assert not controller.success
    assert not controller.failed


def test_unreachable_waypoint_is_skipped_after_a_stall() -> None:
    controller = _after_kill(waypoint_index=1)
    for _ in range(HEART_STALL_FRAMES + 2):
        controller.step(_snap(x=192, y=133))
    assert controller.waypoint_index == 2
    assert any(note.startswith("heart_wp_unreachable") for note in controller.notes)


def test_sweep_budget_exhaustion_fails_closed() -> None:
    controller = _after_kill(sweep_frames=HEART_SWEEP_MAX_FRAMES)
    controller.step(_snap(x=190, y=141))
    assert controller.failed
    assert not controller.success


def test_leaving_the_boss_room_fails() -> None:
    controller = _after_kill()
    controller.step(_snap(x=190, y=141))
    controller.step(_snap(x=16, y=141, screen=0x2B))
    assert controller.failed
    assert not controller.success


def test_factory_never_shares_instances() -> None:
    assert make_level7_aquamentus_heart_controller() is not (
        make_level7_aquamentus_heart_controller()
    )
