from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from retro_harness.nes import nes_idle_action
from snes_oneshot.primitives import FrameAction
from zelda_i.chain import run_controller_stage
from zelda_i.dungeon import (
    DungeonPhase,
    GenericDungeonRoomController,
    ROOM_23_SPEC,
    ROOM_35_SPEC,
    ROOM_33_SPEC,
    ROOM_42_SPEC,
    ROOM_43_SPEC,
    ROOM_44_SPEC,
    ROOM_45_SPEC,
    ROOM_52_SPEC,
    ROOM_53_SPEC,
    ROOM_54_SPEC,
)
from zelda_i.dungeon_ids import object_name, ram_symbol, room_item_name
from zelda_i.dungeon_lab import LabRequest
from zelda_i.dungeon_trace import (
    TraceRecorder,
    first_trace_divergence,
    ram_delta_report,
    write_state_provenance,
)
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_ROOM_ALL_DEAD,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _room_ram(
    *,
    room: int,
    enemy_type: int = 0,
    enemies: int = 0,
    hp: int = 0,
    x: int = 120,
    y: int = 141,
    keys: int = 0,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 1
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_HEALTH] = 0x20
    ram[ADDR_KEYS] = keys
    for slot in range(1, enemies + 1):
        ram[ADDR_OBJ_TYPE + slot] = enemy_type
        ram[ADDR_OBJ_HP + slot] = hp
        ram[ADDR_LINK_X + slot] = 80 + slot * 8
        ram[ADDR_LINK_Y + slot] = 93 + slot * 8
    return ram


def test_room_specs_support_hp_and_type_only_liveness() -> None:
    stalfos = read_snapshot(
        _room_ram(room=0x53, enemy_type=0x2A, enemies=5, hp=0x20)
    )
    keese = read_snapshot(
        _room_ram(room=0x54, enemy_type=0x1B, enemies=8, hp=0)
    )
    assert len(ROOM_53_SPEC.live_enemies(stalfos)) == 5
    assert len(ROOM_54_SPEC.live_enemies(keese)) == 8
    assert ROOM_52_SPEC.expected_enemy_count == 6
    assert ROOM_52_SPEC.entry.direction == "LEFT"
    assert ROOM_42_SPEC.enemy_types == (0x15,)
    assert ROOM_42_SPEC.entry.direction == "UP"
    assert ROOM_43_SPEC.expected_enemy_count == 5
    assert ROOM_33_SPEC.reward.inventory_field == "keys"
    assert ROOM_33_SPEC.combat.engage_distance == 24
    assert ROOM_33_SPEC.combat.attack_phase == 4
    assert ROOM_23_SPEC.enemy_types == (0x06,)
    assert ROOM_23_SPEC.combat.engage_distance == 96
    assert ROOM_23_SPEC.combat.attack_phase == 2
    assert ROOM_44_SPEC.room_item_id == 0x1D
    assert ROOM_44_SPEC.combat.engage_distance == 64
    assert ROOM_44_SPEC.combat.attack_phase == 7
    assert ROOM_45_SPEC.enemy_types == (0x27,)
    assert ROOM_45_SPEC.combat.engage_distance == 80
    assert ROOM_45_SPEC.combat.engage_dominant_axis is True
    assert ROOM_45_SPEC.combat.attack_phase == 0
    assert ROOM_35_SPEC.enemy_types == (0x3D,)


def test_generic_controller_routes_and_clears_type_only_room() -> None:
    controller = GenericDungeonRoomController(ROOM_54_SPEC)
    source = read_snapshot(_room_ram(room=0x53, x=120, y=109))
    action = controller.step(source)
    assert action.reason == "entry_route"

    live_ram = _room_ram(room=0x54, enemy_type=0x1B, enemies=8, hp=0)
    action = controller.step(read_snapshot(live_ram))
    assert controller.phase is DungeonPhase.FIGHT
    assert action.reason.startswith("combat_")
    assert controller.max_live_enemies == 8

    clear_ram = _room_ram(room=0x54, enemies=0)
    clear_ram[ADDR_ROOM_ALL_DEAD] = 20
    action = controller.step(read_snapshot(clear_ram))
    assert controller.success is True
    assert controller.phase is DungeonPhase.DONE
    assert action.reason == "done"


def test_generic_controller_collects_fixed_inventory_reward() -> None:
    controller = GenericDungeonRoomController(ROOM_53_SPEC)
    controller.phase = DungeonPhase.FIGHT
    live_ram = _room_ram(
        room=0x53,
        enemy_type=0x2A,
        enemies=5,
        hp=0x20,
        keys=0,
    )
    controller.step(read_snapshot(live_ram))

    clear_ram = _room_ram(room=0x53, x=88, y=141, keys=0)
    clear_ram[ADDR_ROOM_ALL_DEAD] = 24
    action = controller.step(read_snapshot(clear_ram))
    assert controller.phase is DungeonPhase.COLLECT_REWARD
    assert action.reason == "collect_reward"

    clear_ram[ADDR_KEYS] = 1
    action = controller.step(read_snapshot(clear_ram))
    assert controller.success is True
    assert action.reason == "done"


def test_trace_diff_and_ram_delta_are_symbolic() -> None:
    ram = _room_ram(room=0x54)
    snap = read_snapshot(ram)
    trace = TraceRecorder(tail_frames=1)
    trace.record(
        frame=0,
        phase="FIGHT",
        reason="idle",
        action=nes_idle_action(),
        snap=snap,
    )
    trace.record(
        frame=1,
        phase="FIGHT",
        reason="move",
        action=nes_idle_action(),
        snap=snap,
    )
    assert len(trace.tail) == 1

    left = trace.frames
    right = [dict(left[0]), {**left[1], "reason": "different"}]
    divergence = first_trace_divergence(left, right)
    assert divergence is not None
    assert divergence["index"] == 1
    assert divergence["changed_fields"] == ["reason"]

    after = ram.copy()
    after[ADDR_KEYS] = 1
    after[0x0200] = 7
    report = ram_delta_report(ram, after)
    assert any(row["symbol"] == "keys" for row in report["known"])
    assert any(row["address"] == 0x0200 for row in report["unknown"])


def test_symbolic_registry_keeps_unknowns_explicit() -> None:
    assert object_name(0x1B) == "keese"
    assert room_item_name(0x16) == "unknown_room_item_16"
    assert object_name(0xFE) == "unknown_object_0xfe"
    assert ram_symbol(ADDR_OBJ_TYPE + 2) == "obj_type[2]"


def test_state_provenance_hashes_source_and_output(tmp_path) -> None:
    source = tmp_path / "source.state"
    output = tmp_path / "output.state"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    sidecar = write_state_provenance(
        output,
        source_state_path=source,
        request={"room": 0x54},
        selected_trial={"success": True},
    )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["state_sha256"] != payload["source_state_sha256"]
    assert payload["development_only"] is True


class _Phase(Enum):
    RUNNING = auto()
    DONE = auto()


@dataclass
class _FakeController:
    success: bool = False
    phase: _Phase = _Phase.RUNNING
    frames: int = 0

    def step(self, _snap) -> FrameAction:
        self.frames += 1
        if self.frames == 2:
            self.success = True
            self.phase = _Phase.DONE
        return FrameAction(nes_idle_action(), "fake")

    def report(self) -> dict:
        return {"frames": self.frames}


class _FakeEnv:
    def __init__(self) -> None:
        self.ram = _room_ram(room=0x53)

    def get_ram(self):
        return self.ram

    def step(self, _action):
        return np.zeros((2, 2, 3), dtype=np.uint8), 0.0, False, False, {}


def test_milestone_stage_runner_reuses_standard_loop() -> None:
    controller = _FakeController()
    obs, result = run_controller_stage(
        _FakeEnv(),
        None,
        name="fake",
        controller=controller,
        max_frames=10,
    )
    assert obs.shape == (2, 2, 3)
    assert result.success is True
    assert result.frames == 2


def test_lab_request_is_serializable() -> None:
    request = LabRequest(
        state="Level1Cleared53",
        room_id=0x54,
        alive_rule=None,
    )
    assert request.to_dict()["room_id_hex"] == "0x54"
