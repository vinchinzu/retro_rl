"""Run ledger books: room items left behind, inventory rises, damage rollup. No ROM."""

from __future__ import annotations

import numpy as np

from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_HEALTH,
    ADDR_HEART_PARTIAL,
    ADDR_HELP_DROP_COUNT,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_ROOM_ITEM_ID,
    ADDR_SCREEN,
    ADDR_UW_FLAGS_L1_6,
    ADDR_WORLD_KILL_COUNT,
    PLAY_MODE,
    WORLD_FLAG_ITEM,
)
from zelda_i.spine.ledger import RunLedger


class _Env:
    """``env.step`` plays whatever ``script`` does to RAM for this frame."""

    def __init__(self) -> None:
        self.ram = np.zeros(0x800, dtype=np.uint8)
        self.ram[ADDR_MODE] = PLAY_MODE
        self.ram[ADDR_LEVEL] = 6
        self.ram[ADDR_SCREEN] = 0x19
        self.ram[ADDR_ROOM_ITEM_ID] = 0x17  # the L6 map
        self.ram[ADDR_HEALTH] = 0xAA  # 11 containers, full
        self.ram[ADDR_HEART_PARTIAL] = 0xFF
        self.ram[ADDR_LINK_X] = 120
        self.ram[ADDR_LINK_Y] = 141
        self.script = lambda ram: None

    def get_ram(self) -> np.ndarray:
        return self.ram

    def step(self, action, *args, **kwargs):
        self.script(self.ram)
        return None


def _play(env: _Env, frames: int, script=None) -> None:
    env.script = script or (lambda ram: None)
    for _ in range(frames):
        env.step([0] * 9)


def test_a_room_item_left_behind_is_missed_and_a_taken_one_is_not() -> None:
    env = _Env()
    ledger = RunLedger()
    ledger.attach(env)
    _play(env, 5)

    def to_7a(ram):
        ram[ADDR_SCREEN] = 0x7A
        ram[ADDR_ROOM_ITEM_ID] = 0x19  # a key

    _play(env, 1, to_7a)
    _play(env, 3)
    env.ram[ADDR_UW_FLAGS_L1_6 + 0x7A] |= WORLD_FLAG_ITEM

    def to_78(ram):
        ram[ADDR_SCREEN] = 0x78
        ram[ADDR_ROOM_ITEM_ID] = 0x03  # no item

    _play(env, 1, to_78)
    items = ledger.report(env.ram)["room_items"]
    assert items["rooms"] == 2 and items["taken"] == 1
    assert items["missed"] == [
        {"room": "6:19", "item": "map", "visits": 1, "taken": False}
    ]


def test_inventory_rises_split_play_from_writes_and_ignore_a_load() -> None:
    env = _Env()
    ledger = RunLedger()
    ledger.attach(env)
    _play(env, 2)
    _play(env, 1, lambda ram: ram.__setitem__(ADDR_KEYS, 1))  # picked up
    env.ram[ADDR_BOMBS] = 8  # Survival top-up between frames
    _play(env, 1)
    env.ram[ADDR_LINK_X] = 40  # a save-state load: Link moved, keys jump
    env.ram[ADDR_KEYS] = 5
    _play(env, 1)
    rows = [(g["field"], g["from"], g["to"], g["source"]) for g in ledger.report()["gains"]]
    assert rows == [("keys", 0, 1, "play"), ("bombs", 0, 8, "write")]


def test_rupee_count_up_is_one_row_and_damage_rolls_up_per_room() -> None:
    env = _Env()
    ledger = RunLedger()
    ledger.attach(env)
    _play(env, 1)
    _play(env, 5, lambda ram: ram.__setitem__(0x066D, int(ram[0x066D]) + 1))
    _play(env, 1, lambda ram: ram.__setitem__(ADDR_HEALTH, 0xA8))  # two hearts
    rep = ledger.report()
    assert [(g["field"], g["from"], g["to"]) for g in rep["gains"]] == [("rupees", 0, 5)]
    assert rep["rooms"]["6:19"]["damage"] == 2.0
    assert rep["damage"] == 2.0


def test_boot_ram_before_the_first_play_frame_is_not_damage() -> None:
    env = _Env()
    env.ram[ADDR_MODE] = 0  # title / file select
    env.ram[ADDR_HEALTH] = 0xFF
    ledger = RunLedger()
    ledger.attach(env)
    _play(env, 1)
    _play(env, 1, lambda ram: ram.__setitem__(ADDR_HEALTH, 0x22))
    _play(env, 1, lambda ram: ram.__setitem__(ADDR_MODE, PLAY_MODE))
    assert ledger.report()["damage"] == 0.0


def test_ninth_kill_window_is_recorded_once_and_flags_fairy_priority() -> None:
    env = _Env()
    ledger = RunLedger()
    ledger.attach(env)
    _play(env, 1)
    _play(env, 3, lambda ram: (ram.__setitem__(ADDR_HELP_DROP_COUNT, 9),
                              ram.__setitem__(ADDR_WORLD_KILL_COUNT, 15)))
    windows = ledger.report()["forced_drop_windows"]
    assert windows == [{"frame": 2, "room": "6:19", "world_kills": 15,
                        "fairy_preempts_next_kill": True}]
