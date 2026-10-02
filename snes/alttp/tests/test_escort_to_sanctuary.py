"""Sanctuary arrival is weaker than completed Zelda delivery."""
from dataclasses import asdict, replace
import json

import pytest

from alttp.opening_route.escort_to_sanctuary import evaluate_acceptance
from alttp.ram import (
    DUNGEON_FLOOR,
    LINK_HP,
    LINK_LAYER,
    LINK_X,
    LINK_Y,
    MODULE,
    ROOM_ID,
    SUBMODULE,
    AlttpSnapshot,
    wram_index,
)
from alttp.rollout import PolicyGuard, press
from retro_harness.input_script import FrameAction

import numpy as np


def test_sanctuary_rejects_arrival_and_partial_story() -> None:
    arrived = AlttpSnapshot(
        game_mode=7, submodule=0, room_id=0x12, indoors=True, screen_id=0,
        link_x=1272, link_y=568, link_direction=2, link_action=0,
        camera_x=1152, camera_y=512, dark_world=False, follower=1,
    )
    assert not evaluate_acceptance(arrived, progress=1, progress_flags=0x15)["sanctuary_completed"]
    delivered = replace(arrived, follower=0)
    assert not evaluate_acceptance(delivered, progress=1, progress_flags=0x15)["sanctuary_completed"]
    assert not evaluate_acceptance(delivered, progress=2, progress_flags=0x11)["sanctuary_completed"]
    speaking = replace(delivered, game_mode=14, submodule=2)
    assert not evaluate_acceptance(speaking, progress=2, progress_flags=0x15)["sanctuary_completed"]
    assert evaluate_acceptance(delivered, progress=2, progress_flags=0x15)["sanctuary_completed"]


@pytest.mark.parametrize("field", (LINK_LAYER, DUNGEON_FLOOR))
def test_policy_guard_rejects_indoor_detour_that_changes_layer_or_floor(field: int) -> None:
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[MODULE] = 0x07
    ram[SUBMODULE] = 0
    ram[wram_index(LINK_HP)] = 0x18
    ram[LINK_X:LINK_X + 2] = (100, 0)
    ram[LINK_Y:LINK_Y + 2] = (200, 0)
    ram[ROOM_ID:ROOM_ID + 2] = (0x80, 0)
    ram[LINK_LAYER] = 0
    ram[DUNGEON_FLOOR] = 0xFD

    class Emulator:
        def __init__(self) -> None:
            self.ram = ram.copy()
            self.mask = None

        def get_state(self) -> bytes:
            return b"pin"

        def set_state(self, _state: bytes) -> None:
            self.ram = ram.copy()

        def set_button_mask(self, mask: np.ndarray, _player: int = 0) -> None:
            self.mask = tuple(int(value) for value in mask)

        def step(self) -> None:
            if self.mask == press("UP"):
                self.ram[field] = 1 if field == LINK_LAYER else 0xFE

    class Env:
        def __init__(self) -> None:
            self.em = Emulator()

        def get_ram(self) -> np.ndarray:
            return self.em.ram

    class Inner:
        def step(self, _snapshot: AlttpSnapshot) -> FrameAction:
            return FrameAction(list(press("UP")), "up")

    env = Env()
    guard = PolicyGuard(Inner(), horizon=1, detours=(1,), directions=("UP",))
    guard.bind_env(env)

    hit_frame, left_room = guard._contact((press("UP"),))

    assert env.get_ram()[ROOM_ID] == 0x80  # Room and control stay unchanged.
    assert hit_frame == 1
    assert left_room


@pytest.mark.rom
@pytest.mark.parametrize("offset", (0, 7))
def test_rat_key_replay_predecessor_without_refills(offset: int) -> None:
    """The replay-derived predecessor proves the rat fight and committed replay."""
    from alttp.opening_route.escort_to_sanctuary import _earn_rat_key
    from alttp import primitives
    from alttp.paths import INTEGRATION_DIR
    from alttp.rollout import restores
    from alttp.scripts.replay_tape import _CommittedRecorder
    from alttp.spine.ledger import RunLedger
    from alttp.spine.tape import LiveRun, ram_crc
    from alttp.startup import build_boot_env, no_action, snapshot_env
    from alttp.ram import snapshot_to_diag

    for name in ("rom.sfc", "DevEscort_RatKey.state"):
        if not (INTEGRATION_DIR / name).is_file():
            pytest.skip(f"ALTTP {name} missing")
    env = build_boot_env("DevEscort_RatKey")
    try:
        env.reset()
        recorder = _CommittedRecorder(env)
        live = LiveRun(recorder.attach(), provider="escort_regression")
        start_restores = restores()
        ledger = RunLedger(print_on_close=False)
        ledger._hp = snapshot_env(env).hp
        ledger.attach(live.env)
        for _ in range(offset):
            live.env.step(no_action())
        result = _earn_rat_key(live.env)
        assert result.ok, (result.snapshot, result.reason, ledger.damage_report())
        assert result.snapshot.hp == 16
        assert primitives.sprites_of_type(live.env, (primitives.SPRITE_SMALL_KEY,))
        collected = primitives.collect_nearby(
            live.env, (primitives.SPRITE_SMALL_KEY,), max_distance=400, max_frames=600,
        )
        assert collected.ok
        final = snapshot_env(live.env)
        assert final.num_keys == 1
        assert final.hp == 16
        live.note_rollout_restores(restores() - start_restores)
        assert not live.report().mid_run_state_load
        assert live.report().writes == 0
        assert not ledger.report()["writes"]
    finally:
        env.close()

    controller_audit = asdict(live.report())
    damage = ledger.damage_report()
    buttons, crcs = recorder.buttons, recorder.crcs
    # The recorder retains bound emulator methods after env.close(). Release
    # those references before stable-retro creates the replay core.
    del live, recorder, ledger, env
    import gc
    gc.collect()

    # A pin-only controller green is insufficient: every committed frame
    # must also replay without lookahead, writes or loads from that same pin.
    replay_env = build_boot_env("DevEscort_RatKey")
    try:
        replay_env.reset()
        replay = LiveRun(replay_env, provider="escort_forward_replay")
        for frame, (action, crc) in enumerate(zip(buttons, crcs), 1):
            replay.env.step(action)
            assert ram_crc(replay_env.get_ram()) == crc, f"desync frame {frame}"
        assert snapshot_env(replay_env) == final
        assert replay.report().set_state == replay.report().writes == 0
        print("RAT_KEY " + json.dumps({
            "offset": offset, "frames": len(buttons),
            "final": snapshot_to_diag(final),
            "damage": damage, "controller_audit": controller_audit,
            "replay_audit": asdict(replay.report()), "desynced": 0,
            "source_pin": "DevEscort_RatKey", "development_suffix_only": True,
        }))
    finally:
        replay_env.close()


def test_policy_guard_stops_forecast_at_completed_stage() -> None:
    """A successor stage owns every frame after the mapped goal is reached."""
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[MODULE] = 7
    ram[wram_index(LINK_HP)] = 24
    ram[LINK_X:LINK_X + 2] = (100, 0)
    ram[LINK_Y:LINK_Y + 2] = (200, 0)

    class Emulator:
        def get_state(self) -> bytes:
            return b"goal"

        def set_state(self, _state: bytes) -> None:
            pass

        def set_button_mask(self, _mask, _player) -> None:
            pass

        def step(self) -> None:
            raise AssertionError("forecast stepped beyond the waypoint handoff")

    class Env:
        em = Emulator()

        def get_ram(self):
            return ram

    class CompletedStage:
        success = False

        def step(self, _snapshot):
            self.success = True
            return FrameAction(list(press()), "stage complete")

    guard = PolicyGuard(CompletedStage())
    guard.bind_env(Env())
    assert guard._contact(()) == (None, False)


@pytest.mark.rom
@pytest.mark.parametrize("offset", (0, 7))
def test_no_assist_escort_and_committed_replay(offset: int) -> None:
    """The no-assist rescue predecessor must deliver Zelda, then replay exactly."""
    import gc
    from alttp.opening_route.escort_to_sanctuary import run_from_escort
    from alttp.paths import INTEGRATION_DIR, RECORDINGS_DIR
    from alttp.ram import snapshot_to_diag
    from alttp.rollout import restores
    from alttp.scripts.audit_power_on import story_glance
    from alttp.scripts.replay_tape import _CommittedRecorder
    from alttp.spine.tape import LiveRun, ram_crc
    from alttp.startup import build_boot_env, no_action, snapshot_env
    from retro_harness.env import read_state_bytes

    for name in ("rom.sfc", "DevClean_Zelda.state"):
        if not (INTEGRATION_DIR / name).is_file():
            pytest.skip(f"ALTTP {name} missing")
    pin = read_state_bytes(INTEGRATION_DIR / "DevClean_Zelda.state")
    env = build_boot_env("DevClean_Zelda")
    try:
        env.reset()
        env.em.set_state(pin)
        assert snapshot_env(env).hp == 16
        recorder = _CommittedRecorder(env)
        live = LiveRun(recorder.attach(), provider="no_assist_escort_controller")
        start_restores = restores()
        for _ in range(offset):
            live.env.step(no_action())
        result = run_from_escort(live.env)
        assert result.ok, result.blocker
        final = snapshot_env(env)
        assert final.hp > 0 and story_glance(env)["sanctuary_completed"]
        live.note_rollout_restores(restores() - start_restores)
        assert not live.report().mid_run_state_load
        assert live.report().writes == 0
        controller_audit = asdict(live.report())
        buttons, crcs = recorder.buttons, recorder.crcs
    finally:
        env.close()
    del live, recorder, env
    gc.collect()

    env = build_boot_env("DevClean_Zelda")
    try:
        env.reset()
        env.em.set_state(pin)
        replay = LiveRun(env, provider="no_assist_escort_replay")
        for frame, (action, crc) in enumerate(zip(buttons, crcs), 1):
            replay.env.step(action)
            assert ram_crc(env.get_ram()) == crc, f"desync frame {frame}"
        assert snapshot_env(env) == final
        assert story_glance(env)["sanctuary_completed"]
        assert replay.report().set_state == replay.report().writes == 0
        report = {
            "offset": offset, "frames": len(buttons), "desynced": 0,
            "final": snapshot_to_diag(final), "controller_audit": controller_audit,
            "replay_audit": asdict(replay.report()), "source_pin": "DevClean_Zelda",
            "development_suffix_only": True,
        }
        output = RECORDINGS_DIR / "clean_escape_20261002" / f"escort_offset_{offset}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print("ESCORT " + json.dumps({k: v for k, v in report.items() if k != "controller_audit"}))
    finally:
        env.close()
