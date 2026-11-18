"""Hit log: an HP drop is a hit; an i-frame re-arm is not. No ROM."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alttp.primitives import (
    ANCILLA_TYPE,
    ANCILLA_X_HIGH,
    ANCILLA_X_LOW,
    ANCILLA_Y_HIGH,
    ANCILLA_Y_LOW,
    SPRITE_HP,
    SPRITE_STATE,
    SPRITE_TYPE,
    SPRITE_X_HIGH,
    SPRITE_X_LOW,
    SPRITE_Y_HIGH,
    SPRITE_Y_LOW,
)
from alttp.ram import (
    DARK_WORLD_FLAG,
    DUNGEON_ID,
    INDOORS,
    INVENTORY_FIELDS,
    LINK_ACTION,
    LINK_HP,
    LINK_IFRAMES,
    LINK_KNOCKBACK,
    LINK_X,
    LINK_Y,
    MODULE,
    ROOM_FLAG_CHEST,
    ROOM_ID,
    SAVE_DUNG_INFO,
    SCREEN_ID,
    SUBMODULE,
    wram_index,
)
from alttp.paths import RECORDINGS_DIR
from alttp.spine.ledger import (
    HIT_TRAIL,
    RunLedger,
    Stall,
    census_verified_tip,
    format_soldier_census,
)
from retro_harness.actions import SNES_ACTION_SIZE, snes_action
from retro_harness.input_script import FrameAction


def _u16(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF


def _sprite(
    ram: np.ndarray, slot: int, sprite_type: int, x: int, y: int, hp: int = 4
) -> None:
    ram[SPRITE_STATE + slot] = 9
    ram[SPRITE_TYPE + slot] = sprite_type
    ram[SPRITE_HP + slot] = int(hp) & 0xFF
    ram[SPRITE_X_LOW + slot] = x & 0xFF
    ram[SPRITE_X_HIGH + slot] = (x >> 8) & 0xFF
    ram[SPRITE_Y_LOW + slot] = y & 0xFF
    ram[SPRITE_Y_HIGH + slot] = (y >> 8) & 0xFF


def _poke_items(ram: np.ndarray, items: dict[str, int]) -> None:
    by_name = {field.name: field for field in INVENTORY_FIELDS}
    for name, value in items.items():
        field = by_name[name]
        addr = wram_index(field.offset)
        ram[addr] = int(value) & 0xFF
        if field.size == 2:
            ram[addr + 1] = (int(value) >> 8) & 0xFF


def _bomb(ram: np.ndarray, slot: int, x: int, y: int) -> None:
    ram[ANCILLA_TYPE + slot] = 0x07
    ram[ANCILLA_X_LOW + slot] = x & 0xFF
    ram[ANCILLA_X_HIGH + slot] = (x >> 8) & 0xFF
    ram[ANCILLA_Y_LOW + slot] = y & 0xFF
    ram[ANCILLA_Y_HIGH + slot] = (y >> 8) & 0xFF


class _FakeEnv:
    """Scripted RAM. ``step`` paints the next frame; the ledger reads after."""

    def __init__(self, frames: list[dict]) -> None:
        self.frames = frames
        self.index = 0
        self.ram = np.zeros(0x20000, dtype=np.uint8)

    def get_ram(self) -> np.ndarray:
        return self.ram

    def step(self, action: object) -> None:
        spec = self.frames[self.index]
        self.index += 1
        self.ram[:] = 0
        self.ram[MODULE] = int(spec.get("module", 0x07))
        self.ram[SUBMODULE] = int(spec.get("submodule", 0))
        self.ram[INDOORS] = 1 if spec.get("indoors", True) else 0
        self.ram[SCREEN_ID] = int(spec.get("screen", 0))
        self.ram[DARK_WORLD_FLAG] = 1 if spec.get("dark_world", False) else 0
        self.ram[LINK_KNOCKBACK] = int(spec.get("knockback", 0))
        _u16(self.ram, ROOM_ID, int(spec.get("room", 0x61)))
        _u16(self.ram, LINK_X, int(spec.get("x", 100)))
        _u16(self.ram, LINK_Y, int(spec.get("y", 80)))
        self.ram[LINK_ACTION] = int(spec.get("state", 0))
        self.ram[LINK_IFRAMES] = int(spec.get("iframes", 0))
        self.ram[wram_index(LINK_HP)] = int(spec["hp"]) & 0xFF
        self.ram[DUNGEON_ID] = int(spec.get("dungeon", 0)) & 0xFF
        if "items" in spec:
            _poke_items(self.ram, spec["items"])
        for room in spec.get("open_chests", ()):
            addr = wram_index(SAVE_DUNG_INFO + 2 * int(room))
            self.ram[addr] = int(self.ram[addr]) | (ROOM_FLAG_CHEST & 0xFF)
        if "soldier" in spec:
            sx, sy = spec["soldier"]
            _sprite(self.ram, 0, 0x41, sx, sy)
        for row in spec.get("sprites") or ():
            _sprite(
                self.ram,
                int(row.get("slot", 0)),
                int(row["type"]),
                int(row["x"]),
                int(row["y"]),
                int(row.get("hp", 4)),
            )
        if "bomb" in spec:
            bx, by = spec["bomb"]
            _bomb(self.ram, 0, bx, by)


def test_two_hp_drops_are_two_hits_with_causes() -> None:
    frames: list[dict] = []
    actions = []
    for index in range(24):
        frames.append({"hp": 0x18, "iframes": 40 if index == 9 else 0})
        actions.append(snes_action("RIGHT") if index == 23 else snes_action())
    frames.append({
        "hp": 0x10,
        "state": 2,
        "soldier": (112, 80),
    })
    actions.append(snes_action("LEFT"))
    frames.append({
        "hp": 0x08,
        "state": 2,
        "soldier": (140, 80),
        "bomb": (100, 84),
    })
    actions.append(snes_action("A"))

    env = _FakeEnv(frames)
    ledger = RunLedger()
    ledger.attach(env)
    for action in actions:
        env.step(action)

    assert len(ledger.hits) == 2
    first, second = ledger.hits
    assert first.frame == 25
    assert first.room == 0x61
    assert first.damage == 8
    assert first.cause == "blue soldier"
    assert first.kind == "sprite"
    assert first.distance == 12
    assert first.link_state == 2
    assert len(first.trail) == HIT_TRAIL
    assert first.trail[-1] == ((100, 80), ("LEFT",))
    assert first.trail[-2][1] == ("RIGHT",)

    assert second.frame == 26
    assert second.damage == 8
    assert second.cause == "bomb"
    assert second.kind == "ancilla"
    assert second.distance == 4
    assert second.link_state == 2
    assert second.trail[-1][1] == ("A",)
    assert len(actions[0]) == SNES_ACTION_SIZE

    report = ledger.report()
    assert report["frames"] == 26
    assert [row["cause"] for row in report["hits"]] == ["blue soldier", "bomb"]


def test_scripted_flip_is_one_flutter() -> None:
    # Right twice, then left: one reversal. The knockback shove back to the
    # right is not a second flutter. The next room walks straight.
    frames: list[dict] = [
        {"hp": 0x18, "room": 0x61, "x": 100},
        {"hp": 0x18, "room": 0x61, "x": 101},
        {"hp": 0x18, "room": 0x61, "x": 102},
        {"hp": 0x18, "room": 0x61, "x": 101},
        {"hp": 0x18, "room": 0x61, "x": 110, "knockback": 12},
        {"hp": 0x18, "room": 0x61, "x": 109},
    ]
    frames += [
        {"hp": 0x18, "room": 0x50, "x": 50 + step} for step in range(8)
    ]
    env = _FakeEnv(frames)
    ledger = RunLedger()
    ledger.attach(env)
    for _ in frames:
        env.step(snes_action())

    assert ledger.visits[0].location == ("in", 0x61)
    assert ledger.visits[0].frames == 6
    assert ledger.visits[0].flutters == 1
    assert ledger.visits[1].location == ("in", 0x50)
    assert ledger.visits[1].frames == 8
    assert ledger.visits[1].flutters == 0
    report = ledger.report()
    assert report["flutters"] == 1
    assert report["slowest_visits"][0]["room"] == "in:0050"
    assert report["slowest_visits"][0]["frames"] == 8
    assert report["rooms"]["in:0061"] == {
        "visits": 1,
        "frames": 6,
        "max_visit": 6,
        "flutters": 1,
    }


def test_between_step_write_is_labelled_write() -> None:
    held = {"rupees": 10, "bombs": 5, "keys": 2, "arrows": 4, "magic": 16}
    poked = {"rupees": 25, "bombs": 5, "keys": 1, "arrows": 4, "magic": 16}
    spent = {"rupees": 20, "bombs": 3, "keys": 1, "arrows": 1, "magic": 8}
    frames = [
        {"hp": 0x18, "room": 0x71, "dungeon": 0x02, "items": held},
        {"hp": 0x18, "room": 0x71, "dungeon": 0x02, "items": poked},
        {"hp": 0x10, "room": 0x71, "dungeon": 0x02, "items": spent},
        {
            "hp": 0x10,
            "room": 0,
            "dungeon": 0xFF,
            "indoors": False,
            "module": 0x09,
            "screen": 0x1B,
            "items": spent,
            "open_chests": [0x104],
        },
    ]
    env = _FakeEnv(frames)
    ledger = RunLedger(route_rooms=(0x71, 0x104))
    ledger.attach(env)
    env.step(snes_action())
    _poke_items(env.ram, poked)
    env.step(snes_action())
    env.step(snes_action())
    env.step(snes_action())

    writes = [change for change in ledger.changes if change.source == "write"]
    assert [(change.field, change.before, change.after) for change in writes] == [
        ("rupees", 10, 25),
        ("keys", 2, 1),
    ]
    assert all(change.source == "write" for change in writes)
    assert ledger.report()["writes"][0]["source"] == "write"

    play = {
        change.field: (change.before, change.after, change.source)
        for change in ledger.changes
        if change.source == "play"
    }
    assert play["bombs"] == (5, 3, "play")
    assert play["arrows"] == (4, 1, "play")
    assert play["magic"] == (16, 8, "play")
    assert play["rupees"] == (25, 20, "play")
    assert "hp" not in play
    assert {row["field"] for row in ledger.spends} == {
        "keys",
        "bombs",
        "arrows",
        "magic",
        "rupees",
    }
    assert ledger.missed_chests == [{"dungeon": 0x02, "room": 0x71, "frame": 4}]


def test_bottle_drop_is_a_spend() -> None:
    frames = [
        {"hp": 0x08, "room": 0x55, "items": {"bottle_1": 3}},
        {"hp": 0x18, "room": 0x55, "items": {"bottle_1": 2}},
    ]
    env = _FakeEnv(frames)
    ledger = RunLedger(print_on_close=False)
    ledger.attach(env)
    env.step(snes_action())
    env.step(snes_action())
    assert ledger.spends == [
        {
            "field": "bottle_1",
            "room": "in:0055",
            "frame": 2,
            "from": 3,
            "to": 2,
            "source": "play",
        }
    ]


def test_census_keeps_zero_hp_and_ancilla_and_splits_damage() -> None:
    # Knight stays at hp 0. The bomb is an ancilla. The soldier is the hit.
    frames = [
        {
            "hp": 0x18,
            "room": 0x72,
            "sprites": [
                {"slot": 0, "type": 0x41, "x": 140, "y": 80, "hp": 4},
                {"slot": 1, "type": 0x4B, "x": 200, "y": 80, "hp": 0},
            ],
            "bomb": (100, 90),
        },
        {
            "hp": 0x10,
            "room": 0x72,
            "sprites": [
                {"slot": 0, "type": 0x41, "x": 112, "y": 80, "hp": 4},
                {"slot": 1, "type": 0x4B, "x": 200, "y": 80, "hp": 0},
            ],
            "bomb": (160, 80),
        },
    ]
    env = _FakeEnv(frames)
    ledger = RunLedger(print_on_close=False)
    ledger.attach(env)
    env.step(snes_action())
    ledger.begin_stage("fight")
    env.step(snes_action())

    report = ledger.report()
    room = report["sprites"]["in:0072"]
    assert room["max_alive"] == 3
    by_name = {row["name"]: row for row in room["types"]}
    assert by_name["knight"]["hp0"] == 2
    assert by_name["knight"]["kind"] == "sprite"
    assert by_name["bomb"]["kind"] == "ancilla"
    assert by_name["blue soldier"]["hurt"] == 1
    assert set(report["soldiers_by_room"]["in:0072"]) == {"blue soldier", "knight"}

    damage = report["damage"]
    assert damage["hp_in"] == 0x18
    assert damage["hp_out"] == 0x10
    assert damage["hits_by_cause"] == {"blue soldier": 1}
    assert damage["damage_by_room"] == {"in:0072": 8}
    assert [stage["name"] for stage in damage["stages"]] == ["run", "fight"]
    assert damage["stages"][0]["damage"] == 0
    assert damage["stages"][1]["hp_in"] == 0x18
    assert damage["stages"][1]["hp_out"] == 0x10
    assert damage["stages"][1]["hits_by_cause"] == {"blue soldier": 1}


def test_reason_totals_and_forced_stall_writes_dump(tmp_path) -> None:
    frames = [{"hp": 0x18, "room": 0x61, "x": 100, "y": 80} for _ in range(6)]
    env = _FakeEnv(frames)
    ledger = RunLedger(
        tag="forced",
        watch_stall=True,
        stall_frames=4,
        stall_pixels=4,
        recordings=tmp_path,
        print_on_close=False,
    )
    ledger.attach(env)
    ledger.begin_stage("castle")
    idle = FrameAction(list(snes_action()), "idle")
    with pytest.raises(Stall, match="stall"):
        for _ in range(6):
            env.step(idle)

    assert ledger.blocker == "stall"
    path = tmp_path / "forced_stall.json"
    assert path.is_file()
    dump = json.loads(path.read_text(encoding="utf-8"))
    assert dump["blocker"] == "stall"
    assert len(dump["frames"]) == 4
    assert dump["frames"][-1]["reason"] == "idle"
    assert dump["frames"][-1]["xy"] == [100, 80]
    assert dump["frames"][-1]["buttons"] == []
    assert ledger.report()["reason_by_room"]["in:0061"]["idle"] == 4
    assert ledger.report()["stall"]["blocker"] == "stall"


def test_verified_tip_scan_lists_soldier_types_by_room() -> None:
    report = census_verified_tip(
        RECORDINGS_DIR / "verified_tip.tape.npz",
        RECORDINGS_DIR / "scan" / "index.json",
    )
    assert report["frames_match"] is True
    assert report["tape_frames"] == 14108
    assert report["desynced"] == 0
    assert report["writes"] == 0
    assert report["loads"] == 0
    soldiers = report["soldiers_by_room"]
    assert "knight" in soldiers["in:0055"]
    types = {row["name"]: row for row in report["rooms"]["in:0055"]["types"]}
    assert types["knight"]["hp0"] >= 1
    assert types["knight"]["kind"] == "sprite"
    assert report["rooms"]["in:0055"]["max_alive"] >= 3
    house = {row["name"]: row for row in report["rooms"]["in:0104"]["types"]}
    assert house["uncle and priest"]["hp0"] >= 1
    assert "uncle and priest" not in soldiers["in:0104"]
    text = format_soldier_census(report)
    assert "in:0055: knight" in text
