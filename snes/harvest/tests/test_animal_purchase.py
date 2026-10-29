"""No-ROM gates for the animal-shop cow and chicken buy."""

import numpy as np

from harvest.core.ram_catalog import COW_SLOT_BASE, field_spec, read_ram_value
from harvest.planner.day_phase_catalog import PHASE_SEQUENCES
from harvest.planner.livestock_econ import (
    CHICKEN_PRICE,
    COW_PRICE,
    animal_bought_today,
    purchase_block_reason,
    purchase_grass_required,
    weather_blocks_animal_delivery,
)
from harvest.planner.tasks.sale_menu import (
    CHICKEN_CONFIRM_TEXT_ID,
    GRASS_SHORT_TEXT_ID,
    SHOP_MENU_TEXT_ID,
    AnimalPurchaseSession,
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
    locked[COW_SLOT_BASE] = 1
    for frame in range(1, 180):
        result = receipt.step(world(locked, frame))
        assert result.status == TaskStatus.RUNNING
    result = receipt.step(world(locked, 180))
    assert result.status == TaskStatus.RUNNING
    assert result.reason is not None and "4->5" in result.reason
    assert "7830->2830" in result.reason and "shop=0x24" in result.reason
    assert "slot=0" in result.reason
    for frame in range(181, 3602):
        result = receipt.step(world(locked, frame))
    assert result.status == TaskStatus.FAILURE

    noslot = CowPurchaseTask(species="cow", timeout=3600)
    noslot.reset(world(_shop(), 0))
    free = _shop(money=7830 - COW_PRICE, num_cows=5, input_lock=1)
    result = noslot.step(world(free, 1))
    assert result.status == TaskStatus.RUNNING

    opened = CowPurchaseTask(species="cow", timeout=3600)
    opened.reset(world(_shop(), 0))
    slotted = _shop(money=7830 - COW_PRICE, num_cows=5, input_lock=1)
    slotted[COW_SLOT_BASE] = 1
    result = opened.step(world(slotted, 1))
    assert result.status == TaskStatus.SUCCESS
    assert result.reason is not None and "shop=0x24" in result.reason and "slot=0" in result.reason

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


def test_cow_name_advances_feed_text_before_the_slot_can_succeed():
    """Payment is not the buy. Decline 0x0341, name on 0x5F, then walk."""
    from retro_harness import TaskStatus, WorldState

    from harvest.planner.tasks.chicken_sale import CowPurchaseTask

    def world(ram, frame: int) -> WorldState:
        return WorldState(frame=frame, ram=ram, info={})

    def buttons(result) -> np.ndarray:
        assert result.action is not None
        return result.action.action

    task = CowPurchaseTask(species="cow", timeout=80)
    task.reset(world(_shop(), 0))
    stuck = _shop(money=7830 - COW_PRICE, num_cows=5, input_lock=2)
    _set_text(stuck, 0x0341)
    stuck[0x019B] |= 0x10
    saw_down = False
    for _ in range(30):
        result = task.step(world(stuck, task._step_count + 1))
        assert result.status == TaskStatus.RUNNING
        assert not np.array_equal(buttons(result), make_action(a=True))
        if np.array_equal(buttons(result), make_action(down=True)):
            saw_down = True
    assert saw_down

    stuck[field_spec("dialog_choice_result").address] = 1
    declined = False
    for _ in range(16):
        result = task.step(world(stuck, task._step_count + 1))
        assert result.status == TaskStatus.RUNNING
        if np.array_equal(buttons(result), make_action(a=True)):
            declined = True
            break
    assert declined

    receipt = CowPurchaseTask(species="cow", timeout=80)
    receipt.reset(world(_shop(), 0))
    box = _shop(money=7830 - COW_PRICE, num_cows=5, input_lock=2)
    _set_text(box, 0x0308)
    box[0x019B] |= 0x04
    advanced = False
    for _ in range(12):
        result = receipt.step(world(box, receipt._step_count + 1))
        assert result.status == TaskStatus.RUNNING
        if np.array_equal(buttons(result), make_action(a=True)):
            advanced = True
            break
    assert advanced

    grid = _shop(money=7830 - COW_PRICE, num_cows=5, tilemap=0x5F, input_lock=5)
    named = False
    for _ in range(8):
        result = receipt.step(world(grid, receipt._step_count + 1))
        assert result.status == TaskStatus.RUNNING
        if result.reason and result.reason.startswith("name entry char"):
            named = True
            assert np.array_equal(buttons(result), make_action(a=True))
            break
    assert named

    done = _shop(money=7830 - COW_PRICE, num_cows=5, input_lock=1)
    done[COW_SLOT_BASE] = 1
    result = receipt.step(world(done, receipt._step_count + 1))
    assert result.status == TaskStatus.SUCCESS
    assert result.reason is not None and "slot=0" in result.reason

    still = _shop(money=7830 - COW_PRICE, num_cows=5, input_lock=1)
    result = task.step(world(still, task._step_count + 1))
    assert result.status == TaskStatus.RUNNING


def _armed_name_ram() -> np.ndarray:
    """Fall town after CODE_81E08A. The slot does not exist yet."""
    ram = _shop(tilemap=0x06, money=3800, num_cows=1, planted_grass=33, input_lock=1)
    _set_text(ram, 0x0305)
    ram[0x096E] = 0x42
    ram[0x096F] = 1
    ram[0x019B] = 0
    return ram


def test_armed_name_does_not_walk_and_quotes_a_stuck_machine():
    """$096F=1 is the bought-name machine. 600 idle frames, then stop."""
    from retro_harness import TaskStatus, WorldState

    from harvest.planner.tasks.chicken_sale import CowPurchaseTask

    def world(ram, frame: int) -> WorldState:
        return WorldState(frame=frame, ram=ram, info={})

    def buttons(result) -> np.ndarray:
        assert result.action is not None
        return result.action.action

    stuck = CowPurchaseTask(species="cow", timeout=80)
    ram = _armed_name_ram()
    stuck.reset(world(ram, 0))
    assert stuck._name_armed
    for frame in range(1, 600):
        result = stuck.step(world(ram, frame))
        assert result.status == TaskStatus.RUNNING
        assert np.array_equal(buttons(result), make_action())
    result = stuck.step(world(ram, 600))
    assert result.status == TaskStatus.FAILURE
    assert result.reason is not None
    assert "096F=1" in result.reason
    assert "text=0x0305" in result.reason
    assert "inputstate=1" in result.reason
    assert "0196=0x0000" in result.reason

    live = CowPurchaseTask(species="cow", timeout=80)
    box = _armed_name_ram()
    live.reset(world(box, 0))
    live.step(world(box, 1))
    box[0x096F] = 2
    box[0x019A] = 2
    box[0x019B] |= 0x04
    _set_text(box, 0x030C)
    advanced = False
    for _ in range(12):
        result = live.step(world(box, live._step_count + 1))
        assert result.status == TaskStatus.RUNNING
        pressed = buttons(result)
        assert pressed[4] == pressed[5] == pressed[6] == pressed[7] == 0
        if np.array_equal(pressed, make_action(a=True)):
            advanced = True
            break
    assert advanced

    grid = _armed_name_ram()
    grid[field_spec("tilemap").address] = 0x5F
    grid[0x019A] = 5
    grid[0x096F] = 3
    named = False
    for _ in range(8):
        result = live.step(world(grid, live._step_count + 1))
        assert result.status == TaskStatus.RUNNING
        if result.reason and result.reason.startswith("name entry char"):
            named = True
            assert np.array_equal(buttons(result), make_action(a=True))
            break
    assert named

    done = _armed_name_ram()
    done[COW_SLOT_BASE] = 0x05
    result = live.step(world(done, live._step_count + 1))
    assert result.status == TaskStatus.SUCCESS
    assert result.reason is not None and "slot=0" in result.reason


def test_paid_shop_menu_does_not_press_a_while_script_is_2e():
    """Name bit plus $096E=$2E hands off. A stays off the shop menu."""
    from retro_harness import TaskStatus, WorldState

    from harvest.core.scene import SceneGuard, cow_name_awaits_gate, unowned_scene_press
    from harvest.planner.tasks.chicken_sale import CowPurchaseTask
    from harvest.planner.tasks.navigation import nav_needs_menu_dismiss

    def world(ram, frame: int) -> WorldState:
        return WorldState(frame=frame, ram=ram, info={})

    task = CowPurchaseTask(species="cow", timeout=80)
    task.reset(world(_shop(money=5300, num_cows=1, planted_grass=33), 0))
    paid = _shop(money=300, num_cows=2, planted_grass=33, input_lock=1, tilemap=0x24)
    paid[0x096E] = 0x2E
    _set_u16(paid, 0x11F5A, 0x0002)
    assert cow_name_awaits_gate(paid)
    result = task.step(world(paid, 1))
    assert result.status == TaskStatus.SUCCESS
    assert result.action is None or not np.array_equal(result.action.action, make_action(a=True))

    locked = paid.copy()
    locked[field_spec("input_lock").address] = 2
    receipt = CowPurchaseTask(species="cow", timeout=80)
    receipt.reset(world(_shop(money=5300, num_cows=1, planted_grass=33), 0))
    _set_text(locked, 0x0308)
    locked[0x019B] |= 0x04
    pressed = False
    for _ in range(12):
        result = receipt.step(world(locked, receipt._step_count + 1))
        assert result.status == TaskStatus.RUNNING
        if result.action is not None and np.array_equal(result.action.action, make_action(a=True)):
            pressed = True
            break
    assert pressed

    assert nav_needs_menu_dismiss(paid, 1) is None
    held = nav_needs_menu_dismiss(locked, 1)
    assert held is not None and held.action is not None
    assert not np.array_equal(held.action.action, make_action(a=True))
    assert unowned_scene_press(paid, SceneGuard(), step=4) is None


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
