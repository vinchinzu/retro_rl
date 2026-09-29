"""No-ROM gates for the animal-shop cow and chicken buy."""

import numpy as np

from harvest.core.ram_catalog import field_spec, read_ram_value
from harvest.planner.day_phase_catalog import PHASE_SEQUENCES
from harvest.planner.tasks.sale_menu import (
    CHICKEN_CONFIRM_TEXT_ID,
    CHICKEN_PRICE,
    COW_PRICE,
    GRASS_SHORT_TEXT_ID,
    SHOP_MENU_TEXT_ID,
    AnimalPurchaseSession,
    animal_bought_today,
    purchase_block_reason,
    purchase_grass_required,
    weather_blocks_animal_delivery,
)
from harvest.tasks.nav import make_action


def _ram(**fields: int) -> np.ndarray:
    ram = np.zeros(0x20000, dtype=np.uint8)
    for key, value in fields.items():
        spec = field_spec(key)
        raw = spec.to_storage(value)
        idx = spec.address
        width = spec.width
        for shift in range(width):
            ram[idx + shift] = (raw >> (8 * shift)) & 0xFF
    return ram


def _shop(**fields: int) -> np.ndarray:
    base = {
        "tilemap": 0x24,
        "input_lock": 1,
        "money": 7830,
        "num_cows": 4,
        "num_chickens": 6,
        "planted_grass": 2296,
    }
    base.update(fields)
    return _ram(**base)


def _set_text(ram: np.ndarray, text_id: int) -> None:
    spec = field_spec("dialog_text_id")
    ram[spec.address] = text_id & 0xFF
    ram[spec.address + 1] = (text_id >> 8) & 0xFF


def _set_u16(ram: np.ndarray, address: int, value: int) -> None:
    ram[address] = value & 0xFF
    ram[address + 1] = (value >> 8) & 0xFF


def test_pin_gates_allow_either_animal():
    ram = _shop()
    assert purchase_block_reason(ram, "cow") is None
    assert purchase_block_reason(ram, "chicken") is None
    assert purchase_grass_required(4, 6, "cow") == 176
    assert purchase_grass_required(4, 6, "chicken") == 176


def test_weather_grass_money_cap_and_same_day_block():
    rainy = _shop()
    _set_u16(rainy, 0x0196, 0x0002)
    assert weather_blocks_animal_delivery(rainy)
    assert purchase_block_reason(rainy, "cow") == "weather"

    today = _shop()
    _set_u16(today, 0x11F5A, 0x0020)
    assert animal_bought_today(today)
    assert purchase_block_reason(today, "chicken") == "bought_today"

    assert purchase_block_reason(_shop(money=4999), "cow") == "money"
    assert purchase_block_reason(_shop(money=4999), "chicken") is None
    assert purchase_block_reason(_shop(money=999), "chicken") == "money"
    assert purchase_block_reason(_shop(planted_grass=175), "cow") == "grass"
    assert purchase_block_reason(_shop(num_cows=12), "cow") == "full"
    assert purchase_block_reason(_shop(num_cows=12), "chicken") is None
    assert purchase_block_reason(_shop(num_chickens=12), "chicken") == "full"


def _set_cursor(ram: np.ndarray, cursor: int) -> None:
    ram[field_spec("dialog_choice_result").address] = cursor & 0xFF
    ram[0x019B] |= 0x10


def test_menu_confirms_only_the_highlighted_species():
    cow = AnimalPurchaseSession(species="cow", timeout=80)
    cow_ram = _shop()
    cow.start(cow_ram)
    _set_text(cow_ram, SHOP_MENU_TEXT_ID)
    _set_cursor(cow_ram, 0)
    cow_downs = 0
    cow_a = False
    for _ in range(20):
        decision = cow.step(cow_ram)
        assert decision.status == "run"
        if decision.action is not None and np.array_equal(decision.action, make_action(down=True)):
            cow_downs += 1
        if decision.action is not None and np.array_equal(decision.action, make_action(a=True)):
            cow_a = True
            break
    assert cow_downs == 0 and cow_a

    chicken = AnimalPurchaseSession(species="chicken", timeout=80)
    chicken_ram = _shop()
    chicken.start(chicken_ram)
    _set_text(chicken_ram, SHOP_MENU_TEXT_ID)
    _set_cursor(chicken_ram, 0)
    for _ in range(12):
        decision = chicken.step(chicken_ram)
        assert decision.status == "run"
        assert decision.action is None or not np.array_equal(decision.action, make_action(a=True))
    _set_cursor(chicken_ram, 1)
    saw_a = False
    for _ in range(20):
        decision = chicken.step(chicken_ram)
        assert decision.status == "run"
        if decision.action is not None and np.array_equal(decision.action, make_action(down=True)):
            raise AssertionError("down after the chicken row was highlighted")
        if decision.action is not None and np.array_equal(decision.action, make_action(a=True)):
            saw_a = True
            break
    assert saw_a

    stuck = AnimalPurchaseSession(species="chicken", timeout=80)
    stuck_ram = _shop()
    stuck.start(stuck_ram)
    _set_text(stuck_ram, SHOP_MENU_TEXT_ID)
    _set_cursor(stuck_ram, 0)
    downs = 0
    for _ in range(40):
        decision = stuck.step(stuck_ram)
        assert decision.status == "run"
        if decision.action is not None and np.array_equal(decision.action, make_action(down=True)):
            downs += 1
        assert decision.action is None or not np.array_equal(decision.action, make_action(a=True))
    assert downs >= 11


def test_reject_text_and_wallet_count_are_the_buy():
    ram = _shop()
    session = AnimalPurchaseSession(species="cow", timeout=20)
    session.start(ram)
    _set_text(ram, GRASS_SHORT_TEXT_ID)
    decision = session.step(ram)
    assert decision.status == "failure"
    assert decision.reason == "not enough grass"

    bought = _shop()
    session = AnimalPurchaseSession(species="cow", timeout=20)
    session.start(bought)
    bought_fields = _shop(money=7830 - COW_PRICE, num_cows=5)
    decision = session.step(bought_fields)
    assert decision.status == "success"
    assert "4->5" in decision.reason
    assert f"{7830}->{7830 - COW_PRICE}" in decision.reason

    wrong = _shop()
    session = AnimalPurchaseSession(species="chicken", timeout=20)
    session.start(wrong)
    _set_text(wrong, CHICKEN_CONFIRM_TEXT_ID)
    assert session.step(wrong).status == "run"
    _set_text(wrong, 0x0307)
    _set_cursor(wrong, 0)
    declined = session.step(wrong)
    assert declined.status == "run"
    assert declined.action is not None and np.array_equal(declined.action, make_action(down=True))
    _set_cursor(wrong, 1)
    for _ in range(12):
        session.step(wrong)
    _set_text(wrong, 0x0303)
    assert session.step(wrong).status == "failure"
    assert session.step(wrong).reason == "purchase declined"


def test_registered_buy_waits_for_receipt_and_name_screen_to_close():
    from retro_harness import TaskStatus, WorldState

    from harvest.planner.tasks.chicken_sale import CowPurchaseTask

    def world(ram, frame: int) -> WorldState:
        return WorldState(frame=frame, ram=ram, info={})

    receipt = CowPurchaseTask(species="cow", timeout=3600)
    receipt.reset(world(_shop(), 0))
    locked = _shop(money=7830 - COW_PRICE, num_cows=5, input_lock=0)
    for frame in range(1, 180):
        result = receipt.step(world(locked, frame))
        assert result.status == TaskStatus.RUNNING
    result = receipt.step(world(locked, 180))
    assert result.status == TaskStatus.RUNNING
    assert result.reason is not None and "4->5" in result.reason
    assert "7830->2830" in result.reason
    for frame in range(181, 3602):
        result = receipt.step(world(locked, frame))
    assert result.status == TaskStatus.FAILURE

    opened = CowPurchaseTask(species="cow", timeout=3600)
    opened.reset(world(_shop(), 0))
    free = _shop(money=7830 - COW_PRICE, num_cows=5, input_lock=1)
    result = opened.step(world(free, 1))
    assert result.status == TaskStatus.SUCCESS

    named = CowPurchaseTask(species="chicken", timeout=3600)
    named.reset(world(_shop(), 0))
    grid = _shop(money=7830 - CHICKEN_PRICE, num_chickens=7, tilemap=0x5F, input_lock=5)
    for frame in range(1, 180):
        result = named.step(world(grid, frame))
        assert result.status == TaskStatus.RUNNING
    result = named.step(world(grid, 180))
    assert result.status == TaskStatus.RUNNING
    assert named.purchase_already_registered(grid)
    assert "6->7" in named._bought_reason and "7830->6830" in named._bought_reason
    for frame in range(181, 3602):
        result = named.step(world(grid, frame))
    assert result.status == TaskStatus.FAILURE


def test_counter_sequences_buy_one_species_at_the_sale_counter():
    for name, species in (("buy_cow_counter", "cow"), ("buy_chicken_counter", "chicken")):
        phases = PHASE_SEQUENCES[name]
        assert [phase.phase for phase in phases] == [
            "EXIT_TO_FARM",
            "NAV_ANIMAL_COUNTER",
            phases[2].phase,
        ]
        assert phases[1].params["route"] == "farm_to_animal_shop_counter_sale"
        assert phases[2].kind == "cow_purchase"
        assert phases[2].params["species"] == species
    assert read_ram_value(_shop(), "money") == 7830
